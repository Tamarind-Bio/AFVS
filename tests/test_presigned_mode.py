"""job_storage_mode=presigned: the head signs, the docking job has no AWS
identity and reaches S3 only through presigned URLs.

A fake S3 on loopback serves those URLs and checks each one's signature; the
head side runs against a boto3 stub whose generate_presigned_url mints them.
Docking jobs (templates/afvs_run.py) run as real subprocesses under the test
guard, with boto3 unimportable and only loopback reachable."""

import contextlib
import gzip
import http.server
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import types
import urllib.parse
import uuid
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import pytest

from conftest import (COLLECTIONS, FIXTURES, GUARD_DIR, REPO, TOOLS, FAKE_QVINA, ligand_names,
                      make_collection_tarball, data_key, write_pocket_receptor)

JOB_BUCKET = "job-bucket"
DATA_BUCKET = "data-bucket"
JOB_PREFIX = "scores/someone/dataset"
DATA_PREFIX = "virtual-screen/someone/dataset"
JOB_NAME = "presigned-job"
EXPIRY = 7200
# A collection listed in todo.all that was never staged
MISSING_COLLECTION = "AACAAC_00003"


# --- fake S3 behind presigned URLs ------------------------------------------

class FakeS3:
    """Objects by (bucket, key). A URL is valid only for the method, bucket
    and key it was minted for, and until it is revoked."""

    def __init__(self):
        self.objects = {}
        self.tokens = {}       # token -> (method, bucket, key)
        self.requests = []     # (method, bucket, key, status)
        self.lock = threading.Lock()

        store = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _target(self):
                url = urllib.parse.urlsplit(self.path)
                bucket, key = url.path.lstrip("/").split("/", 1)
                token = urllib.parse.parse_qs(url.query).get("X-Amz-Signature", [""])[0]
                return bucket, urllib.parse.unquote(key), token

            def _reply(self, status, body=b""):
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _handle(self, method):
                bucket, key, token = self._target()
                with store.lock:
                    allowed = store.tokens.get(token) == (method, bucket, key)
                    if not allowed:
                        status = 403
                    elif method == "GET":
                        status = 200 if (bucket, key) in store.objects else 404
                    else:
                        status = 200
                    store.requests.append((method, bucket, key, status))
                if method == "PUT":
                    body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                    if status == 200:
                        with store.lock:
                            store.objects[(bucket, key)] = body
                    self._reply(status)
                else:
                    self._reply(status, store.objects[(bucket, key)] if status == 200 else b"")

            def do_GET(self):
                self._handle("GET")

            def do_PUT(self):
                self._handle("PUT")

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def url(self, method, bucket, key):
        token = f"SECRET{uuid.uuid4().hex}"
        with self.lock:
            self.tokens[token] = (method, bucket, key)
        quoted = urllib.parse.quote(key)
        return f"http://127.0.0.1:{self.server.server_port}/{bucket}/{quoted}?X-Amz-Expires={EXPIRY}&X-Amz-Signature={token}"

    def revoke(self, method, bucket, key):
        with self.lock:
            for token, target in list(self.tokens.items()):
                if target == (method, bucket, key):
                    del self.tokens[token]

    def all_tokens(self):
        return list(self.tokens)

    def close(self):
        self.server.shutdown()


@pytest.fixture
def s3():
    store = FakeS3()
    yield store
    store.close()


# --- boto3 stub for the head --------------------------------------------------

class Head:
    """What the head's boto3 calls did."""

    def __init__(self, store):
        self.store = store
        self.submits = []
        self.presign_calls = []
        self.clients = []
        self.existing_jobs = {}   # job name -> [(job id, createdAt ms)], for ListJobs
        self.fail_next_submit = False
        self.before_submit = None  # called with SubmitJob's arguments

    def client(self, service, config=None, **kwargs):
        head = self
        self.clients.append(service)

        class S3:
            def upload_file(self, filename, bucket, key):
                with open(filename, "rb") as f:
                    head.store.objects[(bucket, key)] = f.read()

            def put_object(self, Bucket, Key, Body, **kwargs):
                head.store.objects[(Bucket, Key)] = Body

            def download_fileobj(self, bucket, key, f):
                f.write(head.store.objects[(bucket, key)])

            def generate_presigned_url(self, ClientMethod, Params, ExpiresIn):
                head.presign_calls.append((ClientMethod, Params['Bucket'], Params['Key'], ExpiresIn))
                method = {'get_object': "GET", 'put_object': "PUT"}[ClientMethod]
                return head.store.url(method, Params['Bucket'], Params['Key'])

        class Batch:
            def submit_job(self, **kwargs):
                if head.before_submit:
                    head.before_submit(kwargs)
                if head.fail_next_submit:
                    error, head.fail_next_submit = head.fail_next_submit, False
                    raise error if isinstance(error, Exception) else RuntimeError("SubmitJob failed")
                head.submits.append(kwargs)
                job_id = f"job-{len(head.submits)}"
                head.existing_jobs.setdefault(kwargs['jobName'], []).append((job_id, int(time.time() * 1000)))
                return {'jobId': job_id}

            def list_jobs(self, jobQueue, filters, nextToken=None):
                # Always paged: an empty first page that only carries a token
                if nextToken is None:
                    return {'jobSummaryList': [], 'nextToken': "page-2"}
                name = filters[0]['values'][0]
                return {'jobSummaryList': [{'jobId': job_id, 'jobName': name, 'createdAt': created}
                                           for job_id, created in head.existing_jobs.get(name, [])]}

            def terminate_job(self, **kwargs):
                raise AssertionError("presigned mode must never terminate jobs")

            def cancel_job(self, **kwargs):
                raise AssertionError("presigned mode must never cancel jobs")

        return {'s3': S3, 'batch': Batch}[service]()


@pytest.fixture
def head(s3, monkeypatch):
    head = Head(s3)

    class Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    boto3 = types.ModuleType("boto3")
    boto3.client = head.client
    botocore = types.ModuleType("botocore")
    botocore.config = types.ModuleType("botocore.config")
    botocore.config.Config = Config
    botocore.exceptions = types.ModuleType("botocore.exceptions")
    botocore.exceptions.ClientError = type("ClientError", (Exception,), {})
    for name, module in [("boto3", boto3), ("botocore", botocore), ("botocore.config", botocore.config),
                         ("botocore.exceptions", botocore.exceptions)]:
        monkeypatch.setitem(sys.modules, name, module)
    return head


# --- a presigned screen ---------------------------------------------------------

CTRL = f"""\
job_name={JOB_NAME}
threads_per_docking=1
threads_to_use=1
program_timeout=60
batchsystem=awsbatch
aws_batch_prefix=vf-sealed
aws_batch_number_of_queues=1
aws_batch_jobdef=vf-jobdef-afvs-sealed
aws_batch_array_job_size=1
aws_ecr_repository_name=unused
aws_region=us-west-2
aws_batch_subjob_vcpus=16
aws_batch_subjob_memory=32000
aws_batch_subjob_timeout=1800
data_storage_mode=s3
job_storage_mode=presigned
presigned_url_expiry_seconds={EXPIRY}
data_collection_addressing_mode=metatranche
data_collection_identifier=fixture
job_addressing_mode=metatranche
object_store_job_bucket={JOB_BUCKET}
object_store_job_prefix={JOB_PREFIX}
object_store_data_bucket={DATA_BUCKET}
object_store_data_collection_prefix={DATA_PREFIX}
summary_formats=parquet,csv.gz
print_attrs_in_summary=smi,heavy_atom_count
collection_list_type=standard
collection_list_file=templates/todo.all
dockings_per_subjob=3
ligand_library_format=pdbqt
dynamic_tranche_filtering=0
dynamic_tranche_filtering_regex=[A-F].................
tempdir_default={{tmp}}
docking_scenario_names=qvina02
docking_scenario_methods=qvina02
docking_scenario_replicas=1
docking_scenario_batchsizes=1
docking_scenario_basefolder=../input-files
prescreen_mode=0
prescreen_ligands_per_tranche=1
run_atom_check=1
"""


class Screen:

    def __init__(self, base, s3, head, monkeypatch, real_engine=False):
        self.base, self.s3, self.head, self.monkeypatch = base, s3, head, monkeypatch
        self.tools = base / "afvs" / "tools"
        shutil.copytree(TOOLS, self.tools, ignore=shutil.ignore_patterns("bin", "docker", "__pycache__"))
        inputs = base / "afvs" / "input-files"
        (inputs / "qvina02").mkdir(parents=True)
        (base / "tmp").mkdir()
        if real_engine:
            # The bundled qvina02, docking into the imatinib pocket of 1IEP
            (inputs / "receptors").mkdir()
            write_pocket_receptor(inputs / "receptors" / "receptor.pdb")
            shutil.copyfile(FIXTURES / "receptor" / "config.txt", inputs / "qvina02" / "config.txt")
            self.fakebin = REPO / "tools" / "bin"
        else:
            (inputs / "qvina02" / "config.txt").write_text("receptor = r.pdb\n")
            self.fakebin = base / "fakebin"
            self.fakebin.mkdir()
            (self.fakebin / "qvina02").write_text(FAKE_QVINA.format(python=sys.executable))
            (self.fakebin / "qvina02").chmod(0o755)

        with open(self.tools / "templates" / "todo.all", "w") as todo:
            for key in COLLECTIONS + [MISSING_COLLECTION]:
                todo.write(f"{key} 3\n")
        for key in COLLECTIONS:
            tarball = base / "staging" / data_key(key)
            make_collection_tarball(key, tarball)
            s3.objects[(DATA_BUCKET, f"{DATA_PREFIX}/{data_key(key)}")] = tarball.read_bytes()
        (self.tools / "templates" / "all.ctrl").write_text(CTRL.format(tmp=base / "tmp"))

    def load(self, script):
        self.monkeypatch.syspath_prepend(str(self.tools))
        name = f"presigned_{Path(script).stem}_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(name, self.tools / script)
        module = importlib.util.module_from_spec(spec)
        self.monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    def prepare(self):
        ret = subprocess.run([sys.executable, "./afvs_prepare_folders.py", "--overwrite", "--skip_errors"],
                             cwd=self.tools, capture_output=True, text=True)
        assert ret.returncode == 0, ret.stdout + ret.stderr
        self.monkeypatch.chdir(self.tools)
        with redirect_stdout(io.StringIO()):
            self.load("afvs_prepare_workunits.py").main()

    def submit(self, start, stop):
        module = self.load("afvs_submit_jobs.py")
        self.monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
        with redirect_stdout(io.StringIO()):
            module.process(module.parse_config("../workflow/config.json"), start, stop)

    def status(self):
        return json.loads((self.tools.parent / "workflow" / "status.json").read_text())

    def run_job(self, submit, subjob_id="0", env_override=None):
        """Run one array child of a submitted job the way Batch would: the job
        definition's command, its environment only (no AWS variables) plus
        the array index Batch sets, boto3 unimportable."""
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("AWS_") and key != "AFVS_WORKUNIT_SUBJOB"}
        env.update({item['name']: item['value'] for item in submit['containerOverrides']['environment']})
        env.update({
            'AWS_BATCH_JOB_ARRAY_INDEX': subjob_id,
            'AFVS_HOME': str(self.tools.parent),
            'AFVS_TOOLS_PATH': str(self.fakebin),
            'PYTHONPATH': str(GUARD_DIR),
            'AFVS_TEST_ALLOW_LOOPBACK': "1",
        })
        env.update(env_override or {})
        return subprocess.run(["bash", str(self.tools / "templates" / "template1.presigned.sh")], cwd="/", env=env,
                              capture_output=True, text=True, timeout=300)

    def assert_no_url_leaked(self, *outputs):
        for output in outputs:
            for token in self.s3.all_tokens():
                assert token not in output
            assert "X-Amz-Signature" not in output


@pytest.fixture
def screen(tmp_path, s3, head, monkeypatch):
    return Screen(tmp_path, s3, head, monkeypatch)


def manifest_of(screen, submit):
    env = {item['name']: item['value'] for item in submit['containerOverrides']['environment']}
    url = urllib.parse.urlsplit(env['AFVS_CONFIG_JOB_URL'])
    bucket, key = url.path.lstrip("/").split("/", 1)
    return json.loads(screen.s3.objects[(bucket, urllib.parse.unquote(key))])


# --- tests ----------------------------------------------------------------------

def test_submit_gives_each_job_one_presigned_url_and_no_credentials(screen):
    screen.prepare()
    screen.submit(1, 3)

    assert len(screen.head.submits) == 3
    for jobline, submit in enumerate(screen.head.submits, start=1):
        assert 'tags' not in submit
        assert submit['jobQueue'] == "vf-sealed-queue1"
        assert submit['jobDefinition'] == "vf-jobdef-afvs-sealed"
        env = {item['name']: item['value'] for item in submit['containerOverrides']['environment']}
        assert set(env) == {'AFVS_RUN_MODE', 'AFVS_JOB_STORAGE_MODE', 'AFVS_VCPUS', 'AFVS_TMP_PATH', 'TMPDIR',
                            'AFVS_RUN_SEQUENTIAL', 'AFVS_WORKUNIT', 'AFVS_CONFIG_JOB_URL'}
        assert env['TMPDIR'] == env['AFVS_TMP_PATH']
        assert env['AFVS_JOB_STORAGE_MODE'] == "presigned" and env['AFVS_WORKUNIT'] == str(jobline)

        manifest = manifest_of(screen, submit)
        assert manifest['version'] == 1 and manifest['workunit'] == str(jobline)
        assert list(manifest['subjobs']) == ["0"]
        prefix = f"{JOB_PREFIX}/{JOB_NAME}"
        assert sorted(manifest['subjobs']['0']['outputs']) == sorted([
            f"{prefix}/summary/{jobline}/0.json.gz",
            f"{prefix}/qvina02/logs/{jobline}/0.tar.gz",
            f"{prefix}/qvina02/parquet/{jobline}/0.parquet",
            f"{prefix}/qvina02/csv/{jobline}/0.csv.gz",
        ])

    # Every URL is signed for the configured lifetime, for the job and data buckets only
    assert {expires for *_, expires in screen.head.presign_calls} == {EXPIRY}
    assert {bucket for _, bucket, _, _ in screen.head.presign_calls} == {JOB_BUCKET, DATA_BUCKET}


def test_submit_keeps_the_job_ids_it_got_and_holds_the_lock(screen):
    """A submit that fails partway keeps the ids of the jobs already
    submitted, so a rerun does not submit them again. Status holds the lock
    for its whole run, so submit takes it too and no stale status write can
    drop those ids."""
    screen.prepare()
    module = screen.load("afvs_submit_jobs.py")
    screen.monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    held = []
    real_lock = module.afvs_presign.recovery_lock

    @contextlib.contextmanager
    def lock():
        with real_lock():
            held.append(True)
            yield
            held.pop()

    screen.monkeypatch.setattr(module.afvs_presign, "recovery_lock", lock)

    def before_submit(kwargs):
        assert held, "SubmitJob outside the lock"
        if kwargs['jobName'].endswith("-2"):
            raise RuntimeError("SubmitJob failed")

    screen.head.before_submit = before_submit
    with pytest.raises(RuntimeError), redirect_stdout(io.StringIO()):
        module.process(module.parse_config("../workflow/config.json"), 1, 3)
    workunits = screen.status()['workunits']
    assert workunits["1"]['job_id'] == "job-1" and 'job_id' not in workunits["2"]

    screen.head.before_submit = lambda kwargs: None
    with redirect_stdout(io.StringIO()):
        module.process(module.parse_config("../workflow/config.json"), 1, 3)
    assert [submit['jobName'] for submit in screen.head.submits] == [
        f"afvs-{JOB_NAME}-1", f"afvs-{JOB_NAME}-2", f"afvs-{JOB_NAME}-3"]


def test_jobs_dock_through_presigned_urls_only(screen):
    screen.prepare()
    screen.submit(1, 3)

    for jobline, submit in enumerate(screen.head.submits, start=1):
        ret = screen.run_job(submit)
        assert ret.returncode == 0, ret.stdout + ret.stderr
        screen.assert_no_url_leaked(ret.stdout, ret.stderr)

        prefix = f"{JOB_PREFIX}/{JOB_NAME}"
        summary = json.loads(gzip.decompress(screen.s3.objects[(JOB_BUCKET, f"{prefix}/summary/{jobline}/0.json.gz")]))
        if jobline <= len(COLLECTIONS):
            assert summary['dockings_status'] == {'success': 3, 'failed': 0}
            scores = pd.read_parquet(io.BytesIO(screen.s3.objects[(JOB_BUCKET, f"{prefix}/qvina02/parquet/{jobline}/0.parquet")]))
            assert sorted(scores['ligand']) == ligand_names(COLLECTIONS[jobline - 1])
            assert (JOB_BUCKET, f"{prefix}/qvina02/logs/{jobline}/0.tar.gz") in screen.s3.objects
        else:
            # Never staged: a failed download, as a missing S3 object is
            assert summary['failed_downloads'] == 1 and summary['total_dockings'] == 0

    # The jobs themselves only ever used presigned URLs that the head minted
    assert all(status in (200, 404) for *_, status in screen.s3.requests)


def test_array_padding_child_exits_cleanly(screen):
    """Batch arrays of one are submitted as two; the extra child finds no subjob."""
    screen.prepare()
    screen.submit(1, 1)
    ret = screen.run_job(screen.head.submits[0], subjob_id="1")
    assert ret.returncode == 0, ret.stdout + ret.stderr


@pytest.mark.parametrize("break_url", ["collection", "every_collection_of_many", "output",
                                       "manifest_unreachable", "manifest_malformed"])
def test_a_failing_url_fails_the_subjob_without_leaking_it(screen, break_url):
    """An expired/rejected URL must fail the subjob (recovery re-signs it),
    and no URL may reach the job's output, its errors or tracebacks."""
    many = [f"AACAAD_{n:05d}" for n in range(1, 31)]
    if break_url == "every_collection_of_many":
        # One subjob of 30 collections: more than the job queues for its one
        # downloader at a time, so the job must notice that downloader died
        (screen.tools / "templates" / "todo.all").write_text("".join(f"{key} 3\n" for key in many))
        ctrl = screen.tools / "templates" / "all.ctrl"
        ctrl.write_text(ctrl.read_text().replace("dockings_per_subjob=3\n", "dockings_per_subjob=90\n"))
    screen.prepare()
    screen.submit(1, 1)
    submit = screen.head.submits[0]
    env_override = None
    if break_url == "every_collection_of_many":
        assert len(manifest_of(screen, submit)['subjobs']['0']['collections']) == 30
        for key in many:
            screen.s3.revoke("GET", DATA_BUCKET, f"{DATA_PREFIX}/{data_key(key)}")
    elif break_url == "collection":
        screen.s3.revoke("GET", DATA_BUCKET, f"{DATA_PREFIX}/{data_key(COLLECTIONS[0])}")
    elif break_url == "output":
        screen.s3.revoke("PUT", JOB_BUCKET, f"{JOB_PREFIX}/{JOB_NAME}/qvina02/parquet/1/0.parquet")
    elif break_url == "manifest_unreachable":
        dead = screen.s3.url("GET", JOB_BUCKET, "nowhere")
        port = urllib.parse.urlsplit(dead).port
        env_override = {'AFVS_CONFIG_JOB_URL': dead.replace(f":{port}/", ":9/")}
    else:
        # urllib quotes the whole path and query (signature included) in the
        # InvalidURL it raises for this one
        bad = screen.s3.url("GET", JOB_BUCKET, "no where")
        env_override = {'AFVS_CONFIG_JOB_URL': bad.replace("no%20where", "no where")}

    ret = screen.run_job(submit, env_override=env_override)
    assert ret.returncode != 0
    screen.assert_no_url_leaked(ret.stdout, ret.stderr)
    assert (JOB_BUCKET, f"{JOB_PREFIX}/{JOB_NAME}/summary/1/0.json.gz") not in screen.s3.objects \
        or break_url == "output"


def test_recovery_rounds_build_on_earlier_rounds(screen):
    screen.prepare()
    screen.submit(1, 3)

    def write_status(update):
        path = screen.tools.parent / "workflow" / "status.json"
        status = json.loads(path.read_text())
        update(status)
        path.write_text(json.dumps(status))

    def mark(status, states):
        for workunit_id, state in states.items():
            status['workunits'][workunit_id]['subjobs']['0']['status'] = state

    write_status(lambda status: mark(status, {"1": "SUCCEEDED", "2": "FAILED", "3": "FAILED"}))

    recover = screen.load("afvs_recover_failed_subjobs.py")
    screen.monkeypatch.setattr(sys, "argv", ["afvs_recover_failed_subjobs.py", "--queue", "vf-queue-ondemand"])
    with redirect_stdout(io.StringIO()):
        assert recover.main() == 0

    recovery_path = screen.tools.parent / "workflow" / "status.recovery.json"
    recovery = json.loads(recovery_path.read_text())['recovery_workunits']
    (round_one, entry), = recovery.items()
    assert round_one == f"{JOB_NAME}-r1-1" and entry['round'] == 1
    assert sorted(mapping['original_workunit_id'] for mapping in entry['recovery_subjobs_mapping'].values()) == ["2", "3"]
    submit = screen.head.submits[-1]
    assert 'tags' not in submit and submit['jobQueue'] == "vf-sealed-queue1"

    # The recovery job docks through its own freshly signed manifest
    ret = screen.run_job(submit)
    assert ret.returncode == 0, ret.stdout + ret.stderr
    screen.assert_no_url_leaked(ret.stdout, ret.stderr)
    assert (JOB_BUCKET, f"{JOB_PREFIX}/{JOB_NAME}/summary/recovery-{round_one}/0.json.gz") in screen.s3.objects

    # Round 1 fixed workunit 2 but not 3: round 2 reruns only workunit 3
    entry_two = recovery[round_one]
    for index, mapping in entry_two['recovery_subjobs_mapping'].items():
        entry_two['subjobs'][index]['status'] = "SUCCEEDED" if mapping['original_workunit_id'] == "2" else "FAILED"
    recovery_path.write_text(json.dumps({'recovery_workunits': recovery}))

    with redirect_stdout(io.StringIO()):
        assert recover.main() == 0
    recovery = json.loads(recovery_path.read_text())['recovery_workunits']
    assert list(recovery) == [round_one, f"{JOB_NAME}-r2-1"]
    round_two = recovery[f"{JOB_NAME}-r2-1"]
    assert round_two['round'] == 2
    assert [mapping['original_workunit_id'] for mapping in round_two['recovery_subjobs_mapping'].values()] == ["3"]

    # Nothing failed any more: no round 3
    round_two['subjobs']['0']['status'] = "SUCCEEDED"
    recovery_path.write_text(json.dumps({'recovery_workunits': recovery}))
    submits = len(screen.head.submits)
    out = io.StringIO()
    with redirect_stdout(out):
        assert recover.main() == 0
    assert "No failed subjobs found!" in out.getvalue() and len(screen.head.submits) == submits


def test_windowed_recovery_continues_without_duplicates(screen):
    """--max-subjobs N submits at most N per call; the next call carries on."""
    screen.prepare()
    screen.submit(1, 3)
    path = screen.tools.parent / "workflow" / "status.json"
    status = json.loads(path.read_text())
    for workunit in status['workunits'].values():
        workunit['subjobs']['0']['status'] = "FAILED"
    path.write_text(json.dumps(status))

    recover = screen.load("afvs_recover_failed_subjobs.py")
    screen.monkeypatch.setattr(sys, "argv", ["afvs_recover_failed_subjobs.py", "--max-subjobs", "2"])
    recovered, remaining = [], []
    for _ in range(3):
        before = len(screen.head.submits)
        out = io.StringIO()
        with redirect_stdout(out):
            assert recover.main() == 0
        remaining.append(int(out.getvalue().rsplit("Remaining failed subjobs: ", 1)[1].split()[0]))
        for submit in screen.head.submits[before:]:
            manifest = manifest_of(screen, submit)
            recovered.append(len(manifest['subjobs']))

    assert remaining == [1, 0, 0]
    assert recovered == [2, 1]
    entries = json.loads((screen.tools.parent / "workflow" / "status.recovery.json").read_text())['recovery_workunits']
    originals = [mapping['original_workunit_id'] for entry in entries.values()
                 for mapping in entry['recovery_subjobs_mapping'].values()]
    assert sorted(originals) == ["1", "2", "3"]  # each once


def fail_all(screen):
    path = screen.tools.parent / "workflow" / "status.json"
    status = json.loads(path.read_text())
    for workunit in status['workunits'].values():
        workunit['subjobs']['0']['status'] = "FAILED"
    path.write_text(json.dumps(status))


def recovery_entries(screen):
    return json.loads((screen.tools.parent / "workflow" / "status.recovery.json").read_text())['recovery_workunits']


def recover_call(screen, recover, *args):
    screen.monkeypatch.setattr(sys, "argv", ["afvs_recover_failed_subjobs.py", *args])
    out = io.StringIO()
    with redirect_stdout(out):
        code = recover.main()
    return code, out.getvalue()


def test_a_subjob_that_keeps_failing_does_not_starve_the_round(screen):
    """--from-round R takes only failures from before round R, so a subjob
    that fails again inside round R waits for round R+1."""
    screen.prepare()
    screen.submit(1, 3)
    fail_all(screen)
    recover = screen.load("afvs_recover_failed_subjobs.py")

    def originals_of_round(round_number):
        return [mapping['original_workunit_id'] for entry in recovery_entries(screen).values()
                if entry['round'] == round_number for mapping in entry['recovery_subjobs_mapping'].values()]

    picked = []
    for _ in range(4):
        code, out = recover_call(screen, recover, "--max-subjobs", "1", "--from-round", "1")
        assert code == 0
        picked.append(out.rsplit("Remaining failed subjobs: ", 1)[1].split()[0])
        # Workunit 1 fails again at once, every time it is retried
        entries = recovery_entries(screen)
        for entry in entries.values():
            for index, mapping in entry['recovery_subjobs_mapping'].items():
                if mapping['original_workunit_id'] == "1":
                    entry['subjobs'][index]['status'] = "FAILED"
        (screen.tools.parent / "workflow" / "status.recovery.json").write_text(json.dumps({'recovery_workunits': entries}))

    # Round 1 retried each starting failure exactly once, the flaky one included
    assert sorted(originals_of_round(1)) == ["1", "2", "3"]
    assert picked == ["2", "1", "0", "0"]
    assert all(entry['round'] == 1 for entry in recovery_entries(screen).values())

    # Its renewed failure is round 2's
    code, out = recover_call(screen, recover, "--max-subjobs", "1", "--from-round", "2")
    assert code == 0 and originals_of_round(2) == ["1"]


@pytest.mark.parametrize("case", ["submitted", "not_listed_yet", "never_submitted", "namesake", "rejected",
                                  "throttled"])
def test_recovery_survives_dying_between_record_and_submit(screen, case):
    """A recovery workunit is recorded before SubmitJob; the next call settles
    a record left without a job id by asking Batch for the job by name."""
    screen.prepare()
    screen.submit(1, 1)
    fail_all(screen)
    recover = screen.load("afvs_recover_failed_subjobs.py")
    recovery_file = screen.tools.parent / "workflow" / "status.recovery.json"

    if case == "submitted":
        # SubmitJob succeeded, then the head died before saving the job id
        recover_call(screen, recover)
        entries = recovery_entries(screen)
        (entry,) = entries.values()
        job_id = entry['job_id']
        entry['job_id'] = None
        recovery_file.write_text(json.dumps({'recovery_workunits': entries}))
    elif case in ("rejected", "throttled"):
        # Batch rejected the SubmitJob (no job to wait for), or throttled a
        # retry whose first attempt may have created it
        client_error = sys.modules["botocore.exceptions"].ClientError
        error = client_error("SubmitJob failed")
        error.response = {'Error': {'Code': "ClientException" if case == "rejected" else "TooManyRequestsException"}}
        screen.head.fail_next_submit = error
        with pytest.raises(client_error):
            recover_call(screen, recover)
        assert [entry.get('abandoned', False) for entry in recovery_entries(screen).values()] == [case == "rejected"]
    else:
        screen.head.fail_next_submit = True
        with pytest.raises(RuntimeError):
            recover_call(screen, recover)
        entries = recovery_entries(screen)
        assert [entry['job_id'] for entry in entries.values()] == [None]
        ((recovery_id, entry),) = entries.items()
        if case in ("never_submitted", "namesake"):
            # Long enough ago that ListJobs would list it by now
            entry['recorded_at'] -= 3600
        if case == "namesake":
            # A job named as this workunit's id alone (an earlier attempt, or
            # another screen's) is not this attempt's job
            screen.head.existing_jobs[f"afvs-r-{recovery_id}"] = [("job-other", 0)]
        recovery_file.write_text(json.dumps({'recovery_workunits': entries}))

    # --report-only changes nothing, and counts the subjob as the real call
    # will act on it: in flight while its record may still be listed, failed
    # once that record is too old (or abandoned)
    before = recovery_file.read_text()
    code, out = recover_call(screen, recover, "--report-only")
    assert code == 0 and recovery_file.read_text() == before
    in_flight = case in ("submitted", "not_listed_yet", "throttled")
    recorded = sum(len(entry['recovery_subjobs_mapping']) for entry in recovery_entries(screen).values())
    assert out.rsplit("Remaining failed subjobs: ", 1)[1].split()[0] == ("0" if in_flight else str(recorded))

    submits = len(screen.head.submits)
    code, out = recover_call(screen, recover)
    assert code == 0
    entries = recovery_entries(screen)
    if case == "submitted":
        # Found by name (past an empty first page): kept, and not submitted again
        assert [entry['job_id'] for entry in entries.values()] == [job_id]
        assert len(screen.head.submits) == submits and "No failed subjobs found!" in out
    elif case in ("not_listed_yet", "throttled"):
        # ListJobs may lag SubmitJob: still pending, its subjob not rerun yet
        assert [entry['job_id'] for entry in entries.values()] == [None]
        assert len(screen.head.submits) == submits and "No failed subjobs found!" in out
    else:
        # Never submitted (or rejected): kept as abandoned (its id is never
        # reused), and the subjob is submitted now under a new id
        (old_id, old), (new_id, new) = entries.items()
        assert old['abandoned'] and old['job_id'] is None
        assert new_id != old_id and new['job_id'] is not None
        assert len(screen.head.submits) == submits + 1
        assert screen.head.submits[-1]['jobName'] == new['job_name'] != old['job_name']


def test_the_job_guard_bites(tmp_path):
    """Docking jobs run under the guard; it must actually block boto3 and any
    network but loopback, or the job-side tests above prove nothing."""
    probe = (
        "import socket\n"
        "try:\n    import boto3\nexcept ImportError:\n    print('boto3 blocked')\n"
        "try:\n    socket.create_connection(('192.0.2.1', 443), timeout=1)\n"
        "except OSError as err:\n    print('network blocked' if 'test guard' in str(err) else err)\n"
    )
    env = {**os.environ, 'PYTHONPATH': str(GUARD_DIR), 'AFVS_TEST_ALLOW_LOOPBACK': "1"}
    ret = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True)
    assert ret.stdout.split("\n")[:2] == ["boto3 blocked", "network blocked"]


def qvina_runs():
    import platform
    try:
        return platform.system() == "Linux" and subprocess.run(
            [str(REPO / "tools" / "bin" / "qvina02"), "--version"], capture_output=True, timeout=60).returncode == 0
    except OSError:
        return False


@pytest.mark.skipif(not qvina_runs(), reason="the bundled qvina02 (x86-64 Linux) cannot run here")
def test_real_qvina02_through_presigned_urls(tmp_path, s3, head, monkeypatch):
    """One real docking job end to end: the bundled qvina02, no AWS identity,
    everything in and out through presigned URLs."""
    screen = Screen(tmp_path, s3, head, monkeypatch, real_engine=True)
    screen.prepare()
    screen.submit(1, 2)

    prefix = f"{JOB_PREFIX}/{JOB_NAME}"
    for jobline, submit in enumerate(screen.head.submits, start=1):
        ret = screen.run_job(submit)
        assert ret.returncode == 0, ret.stdout + ret.stderr
        screen.assert_no_url_leaked(ret.stdout, ret.stderr)

        scores = pd.read_parquet(io.BytesIO(s3.objects[(JOB_BUCKET, f"{prefix}/qvina02/parquet/{jobline}/0.parquet")]))
        assert sorted(scores['ligand']) == ligand_names(COLLECTIONS[jobline - 1])
        assert scores['score_min'].between(-15, 0).all()
        print(scores[['ligand', 'score_min', 'score_average']].to_string(index=False))

        with tarfile.open(fileobj=io.BytesIO(s3.objects[(JOB_BUCKET, f"{prefix}/qvina02/logs/{jobline}/0.tar.gz")])) as tar:
            assert len([m for m in tar.getmembers() if m.name.endswith("output.sdf")]) == 3


def test_killall_refuses_in_presigned_mode(screen):
    screen.prepare()
    ret = subprocess.run([sys.executable, "./afvs_killall_awsbatch.py"], cwd=screen.tools,
                         capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': str(GUARD_DIR)})
    assert ret.returncode == 2 and "does not terminate jobs" in ret.stdout


@pytest.mark.parametrize("old, new, message", [
    ("batchsystem=awsbatch", "batchsystem=slurm\nslurm_template=x",
     "job_storage_mode=presigned requires batchsystem=awsbatch and data_storage_mode=s3"),
    (f"presigned_url_expiry_seconds={EXPIRY}", "presigned_url_expiry_seconds=soon",
     "'presigned_url_expiry_seconds' must be a positive integer"),
    (f"presigned_url_expiry_seconds={EXPIRY}", "presigned_url_expiry_seconds=604801",
     "'presigned_url_expiry_seconds' must be at most 604800"),
    # aws_batch_subjob_timeout=1800: URLs must outlast it plus an hour queued
    (f"presigned_url_expiry_seconds={EXPIRY}", "presigned_url_expiry_seconds=5399",
     "'presigned_url_expiry_seconds' must be at least 'aws_batch_subjob_timeout' + 3600"),
    (f"job_name={JOB_NAME}", "job_name=" + "j" * 101, "'job_name' must be at most 100 characters"),
])
def test_prepare_validates_presigned_configs(screen, old, new, message):
    ctrl = screen.tools / "templates" / "all.ctrl"
    ctrl.write_text(ctrl.read_text().replace(old, new))
    ret = subprocess.run([sys.executable, "./afvs_prepare_folders.py", "--overwrite"], cwd=screen.tools,
                         capture_output=True, text=True)
    assert ret.returncode == 1 and message in ret.stdout
