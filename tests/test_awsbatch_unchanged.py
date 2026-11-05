"""The AWS Batch / S3 paths behave exactly as before presigned mode.

boto3 is replaced by a recorder, and each test pins what the scripts hand to
it (and what they print / write). The expected values were captured from the
scripts at develop 10de6ef; AFVS_TOOLS=<dir> runs the same checks against
another copy of tools/ (that is how the goldens were confirmed on both)."""

import gzip
import importlib.util
import io
import json
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import types
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from conftest import COLLECTIONS, TOOLS as TOOLS_UNDER_TEST


# --- boto3 recorder -------------------------------------------------------

class Recorder:

    def __init__(self):
        self.calls = []
        self.describe_jobs_responses = []
        self.downloads = {}

    def client(self, service, config=None):
        recorder = self

        class Client:
            def submit_job(self, **kwargs):
                recorder.calls.append(('submit_job', kwargs))
                return {'jobId': f"job-{len(recorder.calls)}"}

            def upload_file(self, filename, bucket, key):
                with open(filename, "rb") as f:
                    recorder.calls.append(('upload_file', bucket, key, f.read()))

            def describe_jobs(self, jobs):
                recorder.calls.append(('describe_jobs', list(jobs)))
                return recorder.describe_jobs_responses.pop(0)

            def download_fileobj(self, bucket, key, f):
                recorder.calls.append(('download_fileobj', bucket, key))
                body = recorder.downloads.get((bucket, key))
                if body is None:
                    raise recorder.ClientError(f"NoSuchKey: {key}")
                if isinstance(body, BaseException):
                    raise body
                f.write(body)

        recorder.calls.append(('client', service, getattr(config, 'kwargs', None)))
        return Client()


@pytest.fixture(autouse=True)
def boto(monkeypatch):
    recorder = Recorder()

    class ClientError(Exception):
        pass

    class Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    recorder.ClientError = ClientError
    boto3 = types.ModuleType("boto3")
    boto3.client = recorder.client
    botocore = types.ModuleType("botocore")
    botocore.config = types.ModuleType("botocore.config")
    botocore.config.Config = Config
    botocore.exceptions = types.ModuleType("botocore.exceptions")
    botocore.exceptions.ClientError = ClientError
    for name, module in [("boto3", boto3), ("botocore", botocore), ("botocore.config", botocore.config),
                         ("botocore.exceptions", botocore.exceptions)]:
        monkeypatch.setitem(sys.modules, name, module)
    return recorder


def load(script, monkeypatch):
    """Import a tools/ script as a fresh module (with tools/ importable)."""
    monkeypatch.syspath_prepend(str(TOOLS_UNDER_TEST))
    name = f"under_test_{Path(script).stem}"
    spec = importlib.util.spec_from_file_location(name, TOOLS_UNDER_TEST / script)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


# --- an S3 / AWS Batch screen --------------------------------------------

BATCH_CTRL = """\
job_name=batch-job
threads_per_docking=1
threads_to_use=16
program_timeout=180
batchsystem=awsbatch
aws_batch_prefix=vf
aws_batch_number_of_queues=2
aws_batch_jobdef=vf-jobdef-afvs
aws_batch_array_job_size={array_size}
aws_ecr_repository_name=vf-afvs-ecr
aws_region=us-west-2
aws_batch_subjob_vcpus=16
aws_batch_subjob_memory=32000
aws_batch_subjob_timeout=20800
data_storage_mode=s3
job_storage_mode=s3
data_collection_addressing_mode={addressing}
data_collection_identifier=Enamine_REAL_Space_2022q12-sparse
job_addressing_mode={addressing}
object_store_job_bucket=job-bucket
object_store_job_prefix=scores/someone/dataset
object_store_data_bucket=data-bucket
object_store_data_collection_prefix=virtual-screen/someone/dataset
collection_folder=/home/ec2-user/collections
summary_formats=parquet,csv.gz
print_attrs_in_summary=smi,heavy_atom_count
collection_list_type=standard
collection_list_file=templates/todo.all
dockings_per_subjob=3
ligand_library_format=pdbqt
dynamic_tranche_filtering=0
dynamic_tranche_filtering_regex=[A-F].................
tempdir_default={tmp}
docking_scenario_names=qvina02
docking_scenario_methods=qvina02
docking_scenario_replicas=1
docking_scenario_batchsizes=1
docking_scenario_basefolder=../input-files
prescreen_mode=0
prescreen_ligands_per_tranche=1
run_atom_check=1
"""


def batch_tree(base, addressing="metatranche", array_size=1):
    tools = base / "afvs" / "tools"
    shutil.copytree(TOOLS_UNDER_TEST, tools, ignore=shutil.ignore_patterns("bin", "docker", "__pycache__"))
    (base / "afvs" / "input-files" / "qvina02").mkdir(parents=True)
    (base / "afvs" / "input-files" / "qvina02" / "config.txt").write_text("receptor = r.pdb\n")
    (base / "tmp").mkdir()
    with open(tools / "templates" / "todo.all", "w") as todo:
        for key in COLLECTIONS + ["AACAAC_00003"]:
            todo.write(f"{key} 3\n")
    (tools / "templates" / "all.ctrl").write_text(
        BATCH_CTRL.format(addressing=addressing, array_size=array_size, tmp=base / "tmp"))
    return tools


def prepare_batch(tools, boto, monkeypatch):
    # --skip_errors: the only validation error is the /dev/shm tempdir recommendation
    subprocess.run([sys.executable, "./afvs_prepare_folders.py", "--overwrite", "--skip_errors"], cwd=tools,
                   check=True, capture_output=True)
    monkeypatch.chdir(tools)
    module = load("afvs_prepare_workunits.py", monkeypatch)
    out = io.StringIO()
    with redirect_stdout(out):
        module.main()
    return out.getvalue()


EXPECTED_DATA_KEYS = {
    "metatranche": {
        "AACAAA_00001": "virtual-screen/someone/dataset/AA/AACAAA/00001.tar.gz",
        "AACAAB_00002": "virtual-screen/someone/dataset/AA/AACAAB/00002.tar.gz",
        "AACAAC_00003": "virtual-screen/someone/dataset/AA/AACAAC/00003.tar.gz",
    },
    "hash": {
        "AACAAA_00001": "virtual-screen/someone/dataset/f7/53/Enamine_REAL_Space_2022q12-sparse/pdbqt/AACAAA/0000001.tar.gz",
        "AACAAB_00002": "virtual-screen/someone/dataset/34/4a/Enamine_REAL_Space_2022q12-sparse/pdbqt/AACAAB/0000002.tar.gz",
        "AACAAC_00003": "virtual-screen/someone/dataset/0f/08/Enamine_REAL_Space_2022q12-sparse/pdbqt/AACAAC/0000003.tar.gz",
    },
}

EXPECTED_TASK_KEYS = {
    "metatranche": [f"scores/someone/dataset/batch-job/input/tasks/{i}.tar.gz" for i in (1, 2, 3)],
    "hash": [
        "scores/someone/dataset/92/1f/batch-job/input/tasks/1.tar.gz",
        "scores/someone/dataset/23/94/batch-job/input/tasks/2.tar.gz",
        "scores/someone/dataset/db/ed/batch-job/input/tasks/3.tar.gz",
    ],
}


@pytest.mark.parametrize("addressing", ["metatranche", "hash"])
def test_prepare_workunits_s3_uploads_unchanged(tmp_path, boto, monkeypatch, addressing):
    tools = batch_tree(tmp_path, addressing=addressing)
    out = prepare_batch(tools, boto, monkeypatch)
    assert "Generated 3 workunits" in out

    assert boto.calls[0] == ('client', 's3', {'region_name': 'us-west-2'})
    uploads = [call for call in boto.calls if call[0] == 'upload_file']
    assert [(bucket, key) for _, bucket, key, _ in uploads] == [
        ("job-bucket", key) for key in EXPECTED_TASK_KEYS[addressing]]

    data_keys = {}
    for _, _, _, body in uploads:
        with tarfile.open(fileobj=io.BytesIO(body)) as tar:
            subjobs = json.load(tar.extractfile("vf_input/config.json"))['subjobs']
        assert list(subjobs) == ["0"]
        for key, collection in subjobs["0"]['collections'].items():
            assert collection == {'count': 3, 'mode': 'all', 's3_bucket': 'data-bucket',
                                  's3_download_path': collection['s3_download_path'], 'dockings': 3}
            data_keys[key] = collection['s3_download_path']
    assert data_keys == EXPECTED_DATA_KEYS[addressing]

    status = json.loads((tools.parent / "workflow" / "status.json").read_text())
    assert status == {
        'summary': {'total_dockings': 0, 'docking_succeeded': 0, 'docking_failed': 0, 'skipped_ligands': 0,
                    'vcpu_seconds': 0, 'vcpu_seconds_interrupted': 0, 'failed_downloads': 0,
                    'failed_downloads_dockings': 0},
        'workunits': {str(i): {'subjobs': {'0': {'ligands_expected': 3}}, 's3_download_path': key}
                      for i, key in zip((1, 2, 3), EXPECTED_TASK_KEYS[addressing])},
    }


def test_submit_jobs_awsbatch_arguments_unchanged(tmp_path, boto, monkeypatch):
    tools = batch_tree(tmp_path)
    prepare_batch(tools, boto, monkeypatch)
    boto.calls.clear()

    module = load("afvs_submit_jobs.py", monkeypatch)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    config = module.parse_config("../workflow/config.json")
    with redirect_stdout(io.StringIO()):
        module.process(config, 1, 3)

    def expected(jobline, queue):
        return ('submit_job', {
            'jobName': f'afvs-batch-job-{jobline}',
            'timeout': {'attemptDurationSeconds': 20800},
            'jobQueue': f'vf-queue{queue}',
            'arrayProperties': {'size': 2},
            'jobDefinition': 'vf-jobdef-afvs',
            'containerOverrides': {
                'resourceRequirements': [
                    {'type': 'VCPU', 'value': '16'},
                    {'type': 'MEMORY', 'value': '32000'},
                ],
                'environment': [
                    {'name': 'AFVS_RUN_MODE', 'value': 'awsbatch'},
                    {'name': 'AFVS_JOB_STORAGE_MODE', 'value': 's3'},
                    {'name': 'AFVS_VCPUS', 'value': '16'},
                    {'name': 'AFVS_TMP_PATH', 'value': f"{tmp_path / 'tmp'}/"},
                    {'name': 'AFVS_RUN_SEQUENTIAL', 'value': '0'},
                    {'name': 'AFVS_WORKUNIT', 'value': str(jobline)},
                    {'name': 'AFVS_CONFIG_JOB_OBJECT', 'value': f'scores/someone/dataset/batch-job/input/tasks/{jobline}.tar.gz'},
                    {'name': 'AFVS_CONFIG_JOB_BUCKET', 'value': 'job-bucket'},
                ],
            },
        })

    assert boto.calls == [
        ('client', 'batch', {'region_name': 'us-west-2'}),
        expected(1, 1), expected(2, 2), expected(3, 1),
    ]
    status = json.loads((tools.parent / "workflow" / "status.json").read_text())
    assert [(w['status'], w['job_id'], w['subjobs']['0']['status']) for w in status['workunits'].values()] == [
        ("SUBMITTED", "job-2", "SUBMITTED"), ("SUBMITTED", "job-3", "SUBMITTED"), ("SUBMITTED", "job-4", "SUBMITTED")]


EXPECTED_BATCH_STATUS = (
    "Getting workunit status\n"
    "Getting subjob status\n"
    "Downloading result files \n"
    "\n"
    "-----------------------------------------------------------------\n"
    "AWS Batch Progress\n"
    "-----------------------------------------------------------------\n"
    "\n"
    "Docking count is inaccurate for prescreens. Correct value will\n"
    "be in 'Completed Summary' when finished.\n"
    "\n"
    "         Status   Workunits        Subjobs      Dockings (est.)\n"
    "      SUBMITTED       0               0                 0       \n"
    "        PENDING       0               0                 0       \n"
    "       RUNNABLE       0               0                 0       \n"
    "       STARTING       0               0                 0       \n"
    "        RUNNING       1               1                 3       \n"
    "      SUCCEEDED       1               1                 3       \n"
    "         FAILED       1               1                 3       \n"
    "\n"
    "Active vCPUs: 16\n"
    "\n"
    "-----------------------------------------------------------------\n"
    "Completed Summary\n"
    "-----------------------------------------------------------------\n"
    "\n"
    "* Total Dockings  : 3\n"
    "  - Succeeded     : 2\n"
    "  - Failed        : 1\n"
    "* Skipped ligands : 1\n"
    "* Failed Downloads: 0 (est. 0 dockings)\n"
    "\n"
    "* vCPU seconds per docking: 480.00\n"
    "* vCPU hours total        : 0.40\n"
    "* vCPU hours interrupted  : 0.13\n"
    "\n"
    "-----------------------------------------------------------------\n"
    "Job Timing Analysis (2 completed jobs)\n"
    "-----------------------------------------------------------------\n"
    "* Avg queue wait time : 3.00s (min: 2.00s, max: 4.00s)\n"
    "* Avg execution time  : 45.00s (min: 30.00s, max: 60.00s)\n"
    "* Max execution time  : 60.00s\n"
    "\n"
)


def test_get_status_awsbatch_report_unchanged(tmp_path, boto, monkeypatch):
    tools = batch_tree(tmp_path)
    prepare_batch(tools, boto, monkeypatch)

    status_path = tools.parent / "workflow" / "status.json"
    status = json.loads(status_path.read_text())
    for index, workunit in enumerate(status['workunits'].values(), start=1):
        workunit['status'] = "SUBMITTED"
        workunit['job_id'] = f"j{index}"
    status_path.write_text(json.dumps(status))

    def parent(index, state):
        return {'jobId': f"j{index}", 'status': state, 'arrayProperties': {'statusSummary': {state: 1}}}

    boto.describe_jobs_responses = [
        {'jobs': [parent(1, "SUCCEEDED"), parent(2, "FAILED"), parent(3, "RUNNING")]},
        {'jobs': [
            {'jobId': "j1:0", 'status': "SUCCEEDED", 'createdAt': 1000, 'startedAt': 3000,
             'attempts': [{'startedAt': 3000, 'stoppedAt': 63000}]},
            {'jobId': "j2:0", 'status': "FAILED", 'createdAt': 1000, 'startedAt': 5000,
             'attempts': [{'startedAt': 5000, 'stoppedAt': 35000}, {'startedAt': 40000, 'stoppedAt': 70000}]},
            {'jobId': "j3:0", 'status': "RUNNING"},
        ]},
    ]
    summary = {'total_dockings': 3, 'dockings_status': {'success': 2, 'failed': 1}, 'skipped_ligands': 1,
               'failed_downloads': 0, 'failed_downloads_dockings': 0, 'failed_downloads_log': []}
    boto.downloads[("job-bucket", "scores/someone/dataset/batch-job/summary/1/0.json.gz")] = \
        gzip.compress(json.dumps(summary).encode())

    module = load("afvs_get_status.py", monkeypatch)
    monkeypatch.setattr(sys, "argv", ["afvs_get_status.py"])
    # The downloaders are child processes; fork so they inherit the recorder
    monkeypatch.setattr(module, "Process", multiprocessing.get_context("fork").Process)
    monkeypatch.setattr(module, "Queue", multiprocessing.get_context("fork").Queue)
    monkeypatch.setattr(module.multiprocessing, "cpu_count", lambda: 1)

    out = io.StringIO()
    with redirect_stdout(out):
        module.main()
    assert out.getvalue() == EXPECTED_BATCH_STATUS

    assert boto.calls[-2:] == [('describe_jobs', ["j1", "j2", "j3"]), ('describe_jobs', ["j1:0", "j2:0", "j3:0"])]
    after = json.loads(status_path.read_text())
    assert after['summary']['vcpu_seconds'] == 1440.0
    assert after['summary']['vcpu_seconds_interrupted'] == 480.0


@pytest.fixture
def deadline():
    """Fail a test that hangs instead of hanging the suite."""
    def expired(signum, frame):
        raise AssertionError("timed out")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(60)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)


GOOD_SUMMARY = {'total_dockings': 3, 'dockings_status': {'success': 2, 'failed': 1}, 'skipped_ligands': 1,
                'failed_downloads': 0, 'failed_downloads_dockings': 0, 'failed_downloads_log': []}


@pytest.mark.parametrize("bad", [
    # garbage bytes for one summary, none at all (ClientError) for the other
    {"2": b"\x00not gzip at all", "3": None},
    # gzip'd JSON without the fields, and a summary whose download kills its
    # downloader (so that downloader never sends its stop marker)
    {"2": gzip.compress(json.dumps({'total_dockings': 3}).encode()), "3": SystemExit(3)},
    {"2": gzip.compress(b"[1, 2"), "3": gzip.compress(json.dumps(dict(GOOD_SUMMARY, skipped_ligands="1")).encode())},
])
def test_get_status_s3_bad_summaries_fail_closed(tmp_path, boto, monkeypatch, deadline, bad):
    tools = batch_tree(tmp_path)
    prepare_batch(tools, boto, monkeypatch)

    status_path = tools.parent / "workflow" / "status.json"
    status = json.loads(status_path.read_text())
    for index, workunit in enumerate(status['workunits'].values(), start=1):
        workunit['status'] = "SUBMITTED"
        workunit['job_id'] = f"j{index}"
    status_path.write_text(json.dumps(status))

    boto.describe_jobs_responses = [
        {'jobs': [{'jobId': f"j{index}", 'status': "SUCCEEDED", 'arrayProperties': {'statusSummary': {"SUCCEEDED": 1}}}
                  for index in (1, 2, 3)]},
        {'jobs': [{'jobId': f"j{index}:0", 'status': "SUCCEEDED", 'createdAt': 1000, 'startedAt': 3000,
                   'attempts': [{'startedAt': 3000, 'stoppedAt': 63000}]} for index in (1, 2, 3)]},
    ]
    boto.downloads[("job-bucket", "scores/someone/dataset/batch-job/summary/1/0.json.gz")] = \
        gzip.compress(json.dumps(GOOD_SUMMARY).encode())
    for workunit_id, body in bad.items():
        boto.downloads[("job-bucket", f"scores/someone/dataset/batch-job/summary/{workunit_id}/0.json.gz")] = body

    module = load("afvs_get_status.py", monkeypatch)
    monkeypatch.setattr(sys, "argv", ["afvs_get_status.py"])
    monkeypatch.setattr(module, "Process", multiprocessing.get_context("fork").Process)
    monkeypatch.setattr(module, "Queue", multiprocessing.get_context("fork").Queue)
    monkeypatch.setattr(module.multiprocessing, "cpu_count", lambda: 1)

    out = io.StringIO()
    with redirect_stdout(out):
        module.main()

    lines = out.getvalue().splitlines()
    # The workunits of the bad summaries are reported FAILED too: the wrapper
    # starts recovery from the workunit column
    assert f"{'SUCCEEDED':>15}   {1:^10}   {1:^15}   {3:^15}" in lines
    assert f"{'FAILED':>15}   {2:^10}   {2:^15}   {6:^15}" in lines
    assert "* Total Dockings  : 3\n  - Succeeded     : 2\n  - Failed        : 1\n" \
        "* Skipped ligands : 1\n" in out.getvalue()

    after = json.loads(status_path.read_text())
    assert [w['subjobs']['0']['status'] for w in after['workunits'].values()] == ["SUCCEEDED", "FAILED", "FAILED"]
    # ...but keep the status Batch reported, so a workunit whose array is still
    # running keeps being described for its other subjobs
    assert [w['status'] for w in after['workunits'].values()] == ["SUCCEEDED", "SUCCEEDED", "SUCCEEDED"]
    assert ['overview' in w['subjobs']['0'] for w in after['workunits'].values()] == [True, False, False]
    assert after['summary']['total_dockings'] == 3
    assert after['overview_status']['FAILED']['subjobs'] == 2


def test_afvs_run_s3_output_paths_unchanged(monkeypatch):
    module = load("templates/afvs_run.py", monkeypatch)
    ctx = {'workunit_id': "7", 'subjob_id': "3", 'main_config': {
        'job_storage_mode': "s3", 'object_store_job_prefix': "scores/someone/dataset", 'job_name': "batch-job"}}
    assert module.generate_output_path(ctx, "qvina02", "parquet", "parquet") == \
        "scores/someone/dataset/batch-job/qvina02/parquet/7/3.parquet"
    assert module.generate_output_path(ctx, None, "summary", "json.gz") == \
        "scores/someone/dataset/batch-job/summary/7/3.json.gz"


# --- shell-outs take argv lists --------------------------------------------

HOSTILE_NAME = "lig $(touch INJECTED); touch INJECTED2 'q"


@pytest.fixture
def fake_obabel(tmp_path, monkeypatch):
    """An obabel on PATH that records its argv and writes the -O file."""
    bin_dir = tmp_path / "obabel-bin"
    bin_dir.mkdir()
    record = tmp_path / "obabel-argv.json"
    script = bin_dir / "obabel"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"open({str(record)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "out = sys.argv[sys.argv.index('-O') + 1]\n"
        "open(out, 'w').write('converted')\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.chdir(tmp_path)

    def calls():
        return [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []

    return calls


def test_convert_pose_to_sdf_passes_the_path_as_one_argument(tmp_path, fake_obabel, monkeypatch):
    module = load("templates/afvs_run.py", monkeypatch)
    pose = tmp_path / f"{HOSTILE_NAME}.pdbqt"
    pose.write_text("pose")

    assert module.convert_pose_to_sdf(str(pose)) == f"{pose}.sdf"
    assert fake_obabel() == [["-ipdbqt", str(pose), "-O", f"{pose}.sdf"]]
    assert not list(tmp_path.glob("INJECTED*"))


@pytest.mark.parametrize("script", ["templates/afvs_run.py", "templates/utils.py"])
def test_format_ligand_passes_the_path_as_one_argument(tmp_path, fake_obabel, monkeypatch, script):
    module = load(script, monkeypatch)
    ligand = f"{HOSTILE_NAME}.mol2"
    (tmp_path / ligand).write_text("mol2")

    module.format_ligand(ligand, "pdbqt")
    assert fake_obabel() == [[ligand, "-O", f"{HOSTILE_NAME}pdbqt"]]
    assert not list(tmp_path.glob("INJECTED*"))


def test_convert_ligand_format_passes_the_path_as_one_argument(tmp_path, fake_obabel, monkeypatch):
    module = load("templates/afvs_run.py", monkeypatch)
    ligand = f"{HOSTILE_NAME}.mol2"
    (tmp_path / ligand).write_text("mol2")

    module.convert_ligand_format(ligand, "pdbqt")
    assert fake_obabel() == [[ligand, "-O", f"{HOSTILE_NAME}.pdbqt"]]
    assert not list(tmp_path.glob("INJECTED*"))


def test_missing_obabel_is_not_fatal(tmp_path, monkeypatch):
    module = load("templates/afvs_run.py", monkeypatch)
    monkeypatch.setenv("PATH", str(tmp_path))
    pose = tmp_path / "pose"
    pose.write_text("pose")

    # As before: conversion is skipped and the pose kept under its real extension
    assert module.convert_pose_to_sdf(str(pose)) == f"{pose}.pdbqt"
    module.convert_ligand_format(str(tmp_path / "x.mol2"), "pdbqt")
