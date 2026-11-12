"""The unidock program: one GPU call per batch of ligands, scored like qvina02.

A fake unidock (it reads --ligand_index, writes <stem>_out.pdbqt pose files and
prints nothing else) and a fake obabel stand in for the real binaries, so the
suite needs no GPU. The real Uni-Dock is exercised on a g5 box by the build's
smoke test, not here."""

import gzip
import importlib.util
import io
import json
import os
import queue
import shutil
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path

import pandas as pd
import pytest

from conftest import COLLECTIONS, FIXTURES, TOOLS, ligand_names, make_collection_tarball, data_key, write_pocket_receptor
# The presigned-mode harness: a fake S3 behind presigned URLs, a stubbed head, and a Screen that
# prepares, submits and runs jobs the way Batch does. Its fixtures are used here as they are.
from test_presigned_mode import Screen, JOB_BUCKET, JOB_NAME, JOB_PREFIX, DATA_BUCKET, DATA_PREFIX, head, s3  # noqa: F401


def load(script, monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    name = f"unidock_{Path(script).stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, TOOLS / script)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


FAKE_UNIDOCK = """\
#!{python}
# Fake Uni-Dock: reads the --ligand_index file, writes <stem>_out.pdbqt for each ligand.
#   FAKE_UNIDOCK_LOG    file that gets one json line per call: the ligands, the options, the parent pid
#   FAKE_UNIDOCK_CRASH  comma-separated ligand names; a call that holds one prints an error and exits 1
#   FAKE_UNIDOCK_SKIP   comma-separated ligand names that get no pose file (a call that still exits 0)
import hashlib, json, os, sys
args = sys.argv[1:]
opts = {{args[i][2:]: args[i + 1] for i in range(0, len(args) - 1, 2) if args[i].startswith('--')}}
links = [line.strip() for line in open(opts['ligand_index']) if line.strip()]
names = [os.path.basename(os.path.realpath(link)).split('.')[0] for link in links]
if os.environ.get('FAKE_UNIDOCK_LOG'):
    with open(os.environ['FAKE_UNIDOCK_LOG'], 'a') as log:
        log.write(json.dumps({{'ligands': names, 'opts': opts, 'ppid': os.getppid()}}) + "\\n")
if set(names) & set(os.environ.get('FAKE_UNIDOCK_CRASH', '').split(',')):
    print("CUDA error: an illegal memory access was encountered")
    sys.exit(1)
skip = set(os.environ.get('FAKE_UNIDOCK_SKIP', '').split(','))
for link, name in zip(links, names):
    if name in skip:
        continue
    best = -5 - int(hashlib.sha256(name.encode()).hexdigest()[:4], 16) % 5000 / 1000
    stem = os.path.basename(link).rsplit('.', 1)[0]
    with open(os.path.join(opts['dir'], stem + '_out.pdbqt'), 'w') as out:
        for model, score in enumerate([best, best + 0.5, best + 1.25], start=1):
            out.write(f"MODEL {{model}}\\nREMARK VINA RESULT:   {{score:8.3f}}      0.000      0.000\\n")
            out.write(f"ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C \\nENDMDL\\n")
"""

FAKE_OBABEL = """\
#!{python}
import sys
out = sys.argv[sys.argv.index('-O') + 1]
open(out, 'w').write("sdf of " + open(sys.argv[2]).read()[:20])
"""

GLUE_LIGAND = """\
REMARK  Name = glued
REMARK SMILES C1CCCCCCCCCCC1
ROOT
ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  C   UNL     1       1.500   0.000   0.000  1.00  0.00     0.000 CG0
ATOM      3  C   UNL     1       3.000   0.000   0.000  1.00  0.00     0.000 G0
ENDROOT
TORSDOF 0
"""

CONFIG = """\
receptor = ../input-files/receptors/receptor.pdbqt
center_x = 15.6
center_y = 53.4
center_z = 15.5
size_x = 20
size_y = 20.5
size_z = 22  # a comment
"""


def fake_binary(directory, name, text):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text.format(python=sys.executable))
    path.chmod(0o755)
    return path


# --- pure functions -----------------------------------------------------------

@pytest.mark.parametrize("atom_type, glue", [
    ("G0", True), ("G1", True), ("CG0", True), ("CG12", True), ("G", False), ("CG", False), ("C", False), ("A", False),
    ("NA", False), ("OA", False), ("HD", False), ("Cl", False), ("Br", False), ("Zn", False), ("Ga", False),
])
def test_glue_atom_types(monkeypatch, atom_type, glue):
    module = load("templates/afvs_run.py", monkeypatch)
    assert bool(module.GLUE_ATOM_TYPE.match(atom_type)) is glue


@pytest.mark.parametrize("workers, programs, expected", [
    (16, ["qvina02"], 16),
    (16, ["unidock"], 1),
    (16, ["qvina02", "unidock"], 1),
    (1, ["unidock"], 1),
    (4, ["smina"], 4),
])
def test_a_gpu_program_has_one_docking_worker(monkeypatch, workers, programs, expected):
    module = load("templates/afvs_run.py", monkeypatch)
    ctx = {'vcpus_to_use': workers,
           'main_config': {'docking_scenarios': {f"s{i}": {'program': program} for i, program in enumerate(programs)}}}
    assert module.docking_worker_count(ctx) == expected


def test_the_unidock_entry_is_a_batch_program(monkeypatch):
    module = load("templates/afvs_run.py", monkeypatch)
    entry = module.DOCKING_PROGRAMS['unidock']
    assert entry['ligands'] == "batch"
    assert entry['reject_glue_atoms'] and entry['split_failed_batches'] and entry['max_workers'] == 1
    # qvina02 keeps its own path: single, no guard, no cap
    assert module.DOCKING_PROGRAMS['qvina02'] == {
        'start': module.docking_start_vina, 'end': module.docking_finish_vina, 'ligands': "single"}


@pytest.mark.parametrize("config, gpus", [
    ({}, 0), ({'aws_batch_subjob_gpus': ""}, 0), ({'aws_batch_subjob_gpus': "0"}, 0),
    ({'aws_batch_subjob_gpus': "1"}, 1), ({'aws_batch_subjob_gpus': "2"}, 2),
])
def test_resource_requirements(monkeypatch, config, gpus):
    module = load("afvs_batch_resources.py", monkeypatch)
    expected = [{'type': 'VCPU', 'value': "16"}, {'type': 'MEMORY', 'value': "32000"}]
    if gpus:
        expected.append({'type': 'GPU', 'value': str(gpus)})
    assert module.resource_requirements({'aws_batch_subjob_vcpus': "16", 'aws_batch_subjob_memory': "32000", **config}) == expected


@pytest.mark.parametrize("value", ["-1", "1.5", "one", " 1"])
def test_resource_requirements_refuse_a_bad_gpu_count(monkeypatch, value):
    module = load("afvs_batch_resources.py", monkeypatch)
    with pytest.raises(ValueError):
        module.resource_requirements({'aws_batch_subjob_vcpus': "16", 'aws_batch_subjob_memory': "32000",
                                      'aws_batch_subjob_gpus': value})


# --- the batch path -------------------------------------------------------------

class Batch:
    """A scenario's batch of ligands, ready for docking_process_batch."""

    def __init__(self, tmp_path, monkeypatch, names, config=CONFIG, collections=None):
        self.module = load("templates/afvs_run.py", monkeypatch)
        self.tmp = tmp_path
        self.fakebin = tmp_path / "fakebin"
        fake_binary(self.fakebin, "unidock", FAKE_UNIDOCK)
        fake_binary(self.fakebin, "obabel", FAKE_OBABEL)
        monkeypatch.setenv("PATH", f"{self.fakebin}{os.pathsep}{os.environ['PATH']}")
        self.log = tmp_path / "calls.jsonl"
        monkeypatch.setenv("FAKE_UNIDOCK_LOG", str(self.log))

        self.inputs = tmp_path / "vf_input" / "input-files"
        (self.inputs / "unidock").mkdir(parents=True)
        (self.inputs / "unidock" / "config.txt").write_text(config)
        (self.inputs / "receptors").mkdir()
        (self.inputs / "receptors" / "receptor.pdbqt").write_text("receptor\n")

        self.scenario = {'key': "unidock", 'program': "unidock", 'program_long': "unidock", 'replicas': 1, 'batchsizes': len(names)}
        self.items = []
        self.work = tmp_path / "work"
        for number, name in enumerate(names):
            collection = (collections or {}).get(name, "AACAAA_00001")
            ligand_dir = tmp_path / "collections" / collection
            ligand_dir.mkdir(parents=True, exist_ok=True)
            ligand_path = ligand_dir / f"{name}.pdbqt"
            ligand_path.write_text(f"ligand {name} {collection}\n")
            output_dir = self.work / "output" / "unidock" / name / "0"
            output_dir.mkdir(parents=True)
            self.items.append({
                'ligand_key': name, 'ligand_path': str(ligand_path), 'scenario_key': "unidock",
                'collection_key': collection, 'base_collection_key': collection,
                'config_path': str(self.inputs / "unidock" / "config.txt"), 'program': "unidock", 'program_long': "unidock",
                'input_files_dir': str(self.inputs), 'timeout': 60, 'tools_path': str(self.fakebin),
                'threads_per_docking': 1, 'temp_dir': str(self.work), 'attrs': {'smi': name},
                'output_dir': str(output_dir),
            })

    def run(self):
        summary = queue.Queue()
        self.module.docking_process_batch(summary, self.scenario, self.items, str(self.work))
        done = []
        while not summary.empty():
            done.append(summary.get())
        return done

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []


NAMES = ["benzamide", "ethanol", "phenol", "toluene", "acetone"]


def test_a_batch_is_one_call_and_each_ligand_is_scored_like_qvina02(tmp_path, monkeypatch):
    batch = Batch(tmp_path, monkeypatch, NAMES)
    done = batch.run()

    (call,) = batch.calls()
    assert call['ligands'] == NAMES
    assert call['opts']['search_mode'] == "detail" and call['opts']['scoring'] == "vina" and call['opts']['num_modes'] == "9"
    assert [call['opts'][key] for key in ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")] == [
        "15.6", "53.4", "15.5", "20", "20.5", "22"]
    assert call['opts']['receptor'] == "../input-files/receptors/receptor.pdbqt"
    assert 'seed' not in call['opts']

    (finished,) = done
    assert finished['type'] == "docking_complete" and [item['ligand_key'] for item in finished['items']] == NAMES
    for item in finished['items']:
        assert item['status'] == "success"
        # best of the three poses, all three kept, as docking_finish_vina does
        assert item['score'] == min(item['all_scores']) and len(item['all_scores']) == 3
        assert item['score'] < -5
        # the pose is the ligand's output.sdf, as qvina02's is; the pdbqt is gone
        assert item['output_path'] == f"{item['output_dir']}/output.sdf"
        assert Path(item['output_path']).read_text().startswith("sdf of MODEL 1")
        assert not (Path(item['output_dir']) / "output").exists()
        # which batch docked it
        assert (Path(item['output_dir']) / "dock_uuid").read_text().strip() == finished['uuid']
    # the batch's own log is kept for the archive; its scratch directory is gone
    assert Path(finished['log_path']).exists()
    assert not finished['tmp_run_dir'].exists()


def test_scores_are_the_poses_scores_not_a_neighbours(tmp_path, monkeypatch):
    # Uni-Dock names pose files by number; the right file must reach the right ligand
    batch = Batch(tmp_path, monkeypatch, NAMES)
    (finished,) = batch.run()
    import hashlib
    for item in finished['items']:
        best = -5 - int(hashlib.sha256(item['ligand_key'].encode()).hexdigest()[:4], 16) % 5000 / 1000
        assert item['score'] == pytest.approx(best, abs=1e-3)


def test_ligands_of_the_same_name_in_two_collections_do_not_collide(tmp_path, monkeypatch):
    batch = Batch(tmp_path, monkeypatch, ["ligand"], collections={})
    # a second ligand named "ligand" from another collection, in the same batch
    second = dict(batch.items[0])
    second_dir = tmp_path / "collections" / "AACAAB_00002"
    second_dir.mkdir(parents=True)
    (second_dir / "ligand.pdbqt").write_text("another ligand\n")
    second.update(ligand_path=str(second_dir / "ligand.pdbqt"), collection_key="AACAAB_00002", base_collection_key="AACAAB_00002",
                  output_dir=str(batch.work / "output" / "unidock" / "ligand" / "1"))
    Path(second['output_dir']).mkdir(parents=True)
    batch.items.append(second)

    (finished,) = batch.run()
    assert [item['status'] for item in finished['items']] == ["success", "success"]
    assert batch.calls()[0]['ligands'] == ["ligand", "ligand"]


def test_a_ligand_with_no_pose_fails_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_UNIDOCK_SKIP", "phenol")
    (finished,) = Batch(tmp_path, monkeypatch, NAMES).run()
    status = {item['ligand_key']: item['status'] for item in finished['items']}
    assert status == {name: ("failed" if name == "phenol" else "success") for name in NAMES}
    phenol = next(item for item in finished['items'] if item['ligand_key'] == "phenol")
    assert phenol['log']['reason'] == "Uni-Dock returned no pose"


def test_a_failed_call_is_retried_in_halves_until_the_bad_ligand_fails_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_UNIDOCK_CRASH", "ethanol")
    batch = Batch(tmp_path, monkeypatch, NAMES)
    done = batch.run()

    # 5 -> (2 | 3): the half with ethanol fails again and splits to ethanol alone, the rest are docked
    assert [call['ligands'] for call in batch.calls()] == [
        NAMES,
        ["benzamide", "ethanol"], ["benzamide"], ["ethanol"],
        ["phenol", "toluene", "acetone"],
    ]
    status = {}
    for finished in done:
        for item in finished['items']:
            status[item['ligand_key']] = item['status']
    assert status == {name: ("failed" if name == "ethanol" else "success") for name in NAMES}
    ethanol = next(item for finished in done for item in finished['items'] if item['ligand_key'] == "ethanol")
    assert ethanol['log']['reason'] == "Non zero return code for batched execution"
    # every failed call stays in the report with no ligands, so its log is kept
    assert sum(1 for finished in done if finished['items'] == []) == 2
    # no ligand is reported twice
    reported = [item['ligand_key'] for finished in done for item in finished['items']]
    assert sorted(reported) == sorted(NAMES)
    # each report has its own log directory
    assert len({str(finished['output_dir']) for finished in done}) == len(done)


def test_a_failed_call_of_one_ligand_fails_it_without_looping(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_UNIDOCK_CRASH", "ethanol")
    batch = Batch(tmp_path, monkeypatch, ["ethanol"])
    (finished,) = batch.run()
    assert len(batch.calls()) == 1
    assert [item['status'] for item in finished['items']] == ["failed"]


@pytest.mark.parametrize("config, message", [
    (CONFIG.replace("receptor.pdbqt", "receptor.pdb"), "PDBQT"),
    (CONFIG.replace("center_y = 53.4\n", ""), "center_y"),
    (CONFIG.replace("size_z = 22  # a comment", "size_z = wide"), "size_z"),
    (CONFIG + "search_mode = exhaustive\n", "search_mode"),
    (CONFIG + "num_modes = 0\n", "num_modes"),
    (CONFIG + "num_modes = many\n", "num_modes"),
    (CONFIG + "seed = x\n", "seed"),
])
def test_a_bad_config_is_refused_before_any_call(tmp_path, monkeypatch, config, message):
    batch = Batch(tmp_path, monkeypatch, NAMES, config=config)
    with pytest.raises(RuntimeError, match=message):
        batch.run()
    assert batch.calls() == []


def test_search_mode_modes_and_seed_reach_the_call(tmp_path, monkeypatch):
    batch = Batch(tmp_path, monkeypatch, NAMES, config=CONFIG + "search_mode = balance\nnum_modes = 3\nseed = 7\n")
    batch.run()
    opts = batch.calls()[0]['opts']
    assert (opts['search_mode'], opts['num_modes'], opts['seed']) == ("balance", "3", "7")


# --- a whole job ------------------------------------------------------------------

class UniScreen(Screen):
    """The presigned screen, docking with unidock on a GPU class: batches of 2, four workers asked for
    (so the cap is what leaves one), a GPU requested, and a glued ligand staged in the first collection."""

    def __init__(self, base, s3, head, monkeypatch, real_engine=False):
        super().__init__(base, s3, head, monkeypatch)
        inputs = base / "afvs" / "input-files"
        shutil.rmtree(inputs / "qvina02")
        (inputs / "unidock").mkdir()
        (inputs / "receptors").mkdir()
        if real_engine:
            # The Uni-Dock in the tools' bin (the GPU image has it there), docking into the imatinib
            # pocket of 1IEP with a receptor PDBQT that Open Babel makes, hydrogens added
            self.fakebin = unidock_binary().parent
            write_pocket_receptor(inputs / "receptors" / "receptor.pdb")
            subprocess.run(["obabel", "-ipdb", str(inputs / "receptors" / "receptor.pdb"), "-opdbqt", "-xr", "-h",
                            "-O", str(inputs / "receptors" / "receptor.pdbqt")], check=True, capture_output=True)
            (inputs / "unidock" / "config.txt").write_text(
                (FIXTURES / "receptor" / "config.txt").read_text().replace("receptor.pdb", "receptor.pdbqt"))
        else:
            (inputs / "unidock" / "config.txt").write_text(CONFIG)
            (inputs / "receptors" / "receptor.pdbqt").write_text("receptor\n")
            fake_binary(self.fakebin, "unidock", FAKE_UNIDOCK)
            fake_binary(self.fakebin, "obabel", FAKE_OBABEL)

        ctrl = self.tools / "templates" / "all.ctrl"
        text = ctrl.read_text()
        for old, new in [("docking_scenario_names=qvina02", "docking_scenario_names=unidock"),
                         ("docking_scenario_methods=qvina02", "docking_scenario_methods=unidock"),
                         ("docking_scenario_batchsizes=1", "docking_scenario_batchsizes=2"),
                         ("threads_to_use=1", "threads_to_use=4"),
                         ("aws_batch_prefix=vf-sealed", "aws_batch_prefix=vf-sealed-gpu1"),
                         ("aws_batch_jobdef=vf-jobdef-afvs-sealed", "aws_batch_jobdef=vf-jobdef-afvs-sealed-gpu1")]:
            assert old in text
            text = text.replace(old, new)
        ctrl.write_text(text + "aws_batch_subjob_gpus=1\n")

        # The first collection also holds a ligand with Meeko glue atoms
        glued = base / "staging" / "glued"
        make_collection_tarball(COLLECTIONS[0], glued / "c.tar.gz")
        number = COLLECTIONS[0].split("_", 1)[1]
        with tarfile.open(glued / "c.tar.gz", "r:gz") as source:
            members = [(member, source.extractfile(member).read()) for member in source.getmembers()]
        with tarfile.open(glued / "c2.tar.gz", "w:gz") as tar:
            for member, body in members:
                tar.addfile(member, io.BytesIO(body))
            info = tarfile.TarInfo(f"{number}/glued.pdbqt")
            info.size = len(GLUE_LIGAND.encode())
            tar.addfile(info, io.BytesIO(GLUE_LIGAND.encode()))
        s3.objects[(DATA_BUCKET, f"{DATA_PREFIX}/{data_key(COLLECTIONS[0])}")] = (glued / "c2.tar.gz").read_bytes()

    def job_env(self):
        # The fake engines are found first on PATH (a real engine's directory is not put on it)
        path = f"{self.fakebin}{os.pathsep}{os.environ['PATH']}" if self.fakebin.name == "fakebin" else os.environ['PATH']
        return {'PATH': path, 'FAKE_UNIDOCK_LOG': str(self.tools.parent / "calls.jsonl")}

    def calls(self):
        path = self.tools.parent / "calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def uni(tmp_path, s3, head, monkeypatch):
    return UniScreen(tmp_path, s3, head, monkeypatch)


def test_a_gpu_job_docks_in_batches_on_one_worker_and_skips_glued_ligands(uni):
    uni.prepare()
    uni.submit(1, 2)

    # The job asks for a GPU as well as vCPUs and memory
    for submit in uni.head.submits:
        assert {'type': 'GPU', 'value': "1"} in submit['containerOverrides']['resourceRequirements']
        assert submit['jobQueue'] == "vf-sealed-gpu1-queue1" and submit['jobDefinition'] == "vf-jobdef-afvs-sealed-gpu1"

    prefix = f"{JOB_PREFIX}/{JOB_NAME}"
    seen = 0
    for jobline, submit in enumerate(uni.head.submits, start=1):
        ret = uni.run_job(submit, env_override=uni.job_env())
        assert ret.returncode == 0, ret.stdout + ret.stderr
        uni.assert_no_url_leaked(ret.stdout, ret.stderr)

        # Its three ligands went to Uni-Dock in batches of up to two, every call from the one docking
        # worker the program is capped to (the job asked for four workers)
        job_calls, seen = uni.calls()[seen:], len(uni.calls())
        assert sorted(ligand for call in job_calls for ligand in call['ligands']) == ligand_names(COLLECTIONS[jobline - 1])
        assert all(len(call['ligands']) <= 2 for call in job_calls) and len(job_calls) == 2
        assert len({call['ppid'] for call in job_calls}) == 1

        summary = json.loads(gzip.decompress(uni.s3.objects[(JOB_BUCKET, f"{prefix}/summary/{jobline}/0.json.gz")]))
        scores = pd.read_parquet(io.BytesIO(uni.s3.objects[(JOB_BUCKET, f"{prefix}/unidock/parquet/{jobline}/0.parquet")]))
        assert sorted(scores['ligand']) == ligand_names(COLLECTIONS[jobline - 1])
        assert summary['dockings_status'] == {'success': 3, 'failed': 0}
        assert (scores['score_min'] < 0).all() and (scores['score_min'] <= scores['score_average']).all()
        assert (JOB_BUCKET, f"{prefix}/unidock/logs/{jobline}/0.tar.gz") in uni.s3.objects

        if jobline == 1:
            # The glued ligand is counted and named, never docked
            assert summary['skipped_ligands'] == 1
            assert [(skip['ligand_key'], skip['reason'], skip['scenario']) for skip in summary['skipped_ligand_list']] == [
                ("glued", "macrocycle_glue_atoms", "unidock")]
        else:
            assert summary['skipped_ligands'] == 0

    assert all("glued" not in call['ligands'] for call in uni.calls())


def test_the_poses_are_in_the_logs_archive_as_output_sdf(uni):
    uni.prepare()
    uni.submit(1, 1)
    ret = uni.run_job(uni.head.submits[0], env_override=uni.job_env())
    assert ret.returncode == 0, ret.stdout + ret.stderr

    prefix = f"{JOB_PREFIX}/{JOB_NAME}"
    archive = uni.s3.objects[(JOB_BUCKET, f"{prefix}/unidock/logs/1/0.tar.gz")]
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        names = tar.getnames()
    for ligand in ligand_names(COLLECTIONS[0]):
        assert f"0/{ligand}/0/output.sdf" in names
        assert f"0/{ligand}/0/dock_uuid" in names
    assert not any(name.endswith("/output") for name in names)
    assert any("/batch_exec/" in name for name in names)
    assert not any("glued" in name for name in names)


def unidock_binary():
    return Path(os.environ.get("AFVS_TOOLS", TOOLS)) / "bin" / "unidock"


def unidock_runs():
    import platform
    try:
        return (platform.system() == "Linux" and shutil.which("obabel") is not None and unidock_binary().exists()
                and subprocess.run([str(unidock_binary()), "--version"], capture_output=True, timeout=60).returncode == 0
                and subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=60).returncode == 0)
    except OSError:
        return False


@pytest.mark.skipif(not unidock_runs(), reason="needs Uni-Dock in the tools' bin (x86-64 Linux), a GPU and Open Babel")
def test_real_unidock_through_presigned_urls(tmp_path, s3, head, monkeypatch):
    """One real GPU docking job end to end (AFVS_TOOLS=/opt/AFVS/tools inside the gpu image): the
    real Uni-Dock in batches, no AWS identity, a glued ligand skipped, poses converted by Open Babel."""
    screen = UniScreen(tmp_path, s3, head, monkeypatch, real_engine=True)
    screen.prepare()
    screen.submit(1, 2)

    prefix = f"{JOB_PREFIX}/{JOB_NAME}"
    for jobline, submit in enumerate(screen.head.submits, start=1):
        ret = screen.run_job(submit)
        assert ret.returncode == 0, ret.stdout + ret.stderr
        screen.assert_no_url_leaked(ret.stdout, ret.stderr)

        summary = json.loads(gzip.decompress(s3.objects[(JOB_BUCKET, f"{prefix}/summary/{jobline}/0.json.gz")]))
        scores = pd.read_parquet(io.BytesIO(s3.objects[(JOB_BUCKET, f"{prefix}/unidock/parquet/{jobline}/0.parquet")]))
        assert sorted(scores['ligand']) == ligand_names(COLLECTIONS[jobline - 1])
        assert summary['dockings_status'] == {'success': 3, 'failed': 0}
        assert scores['score_min'].between(-15, 0).all() and (scores['score_min'] <= scores['score_average']).all()
        assert summary['skipped_ligands'] == (1 if jobline == 1 else 0)
        print(scores[['ligand', 'score_min', 'score_average']].to_string(index=False))

        with tarfile.open(fileobj=io.BytesIO(s3.objects[(JOB_BUCKET, f"{prefix}/unidock/logs/{jobline}/0.tar.gz")])) as tar:
            poses = [member for member in tar.getmembers() if member.name.endswith("output.sdf")]
            assert len(poses) == 3
            for pose in poses:
                assert tar.extractfile(pose).read().count(b"$$$$") >= 1
