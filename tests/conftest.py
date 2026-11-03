"""Harness for the local-backend tests.

Every test gets its own copy of the AFVS tree (the tools resolve everything
relative to tools/, e.g. ../workflow), a local_data_root staged with the
fixture collections, and a fake docking engine that prints qvina02-style
output. The real engine is used only by test_real_docking.py.
"""

import gzip
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
# AFVS_TOOLS=<dir> runs the suite against another copy of tools/ (e.g. develop's)
TOOLS = Path(os.environ.get("AFVS_TOOLS", REPO / "tools"))
FIXTURES = Path(__file__).resolve().parent / "fixtures"
GUARD_DIR = Path(__file__).resolve().parent / "guard"

# The fixture: two collections of three ligands each (ligand names are the
# file names under fixtures/ligands/<collection_key>/)
COLLECTIONS = ["AACAAA_00001", "AACAAB_00002"]


def ligand_names(collection_key):
    return sorted(p.stem for p in (FIXTURES / "ligands" / collection_key).glob("*.pdbqt"))


ALL_CTRL = """\
job_name={job_name}
threads_per_docking=1
threads_to_use={threads}
program_timeout=60
batchsystem=local
local_workers={workers}
local_array_job_size={array_size}
data_storage_mode=local
job_storage_mode=local
local_job_root={job_root}
local_data_root={data_root}
data_collection_addressing_mode=metatranche
data_collection_identifier=fixture
job_addressing_mode=metatranche
summary_formats=parquet,csv.gz
print_attrs_in_summary=smi,heavy_atom_count
collection_list_type=standard
collection_list_file=templates/todo.all
dockings_per_subjob={dockings_per_subjob}
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

FAKE_QVINA = """\
#!{python}
# Fake qvina02: prints a vina-style result table for the ligand.
#   FAKE_DOCK_SLEEP     seconds to sleep per docking
#   FAKE_DOCK_FAIL      comma-separated ligand names that exit 1
#   FAKE_DOCK_CRASH     ligand name that kills its whole afvs_run.py (a hard subjob failure)
#   FAKE_DOCK_LOG       file that gets "<ligand> <start> <end>" per docking run
import hashlib, os, shutil, signal, sys, time
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
ligand = os.path.basename(args['--ligand']).split('.')[0]
start = time.time()
time.sleep(float(os.environ.get('FAKE_DOCK_SLEEP', '0')))
if ligand == os.environ.get('FAKE_DOCK_CRASH'):
    os.killpg(os.getpgrp(), signal.SIGKILL)
if os.environ.get('FAKE_DOCK_LOG'):
    with open(os.environ['FAKE_DOCK_LOG'], 'a') as log:
        log.write(f"{{ligand}} {{start}} {{time.time()}}\\n")
if ligand in os.environ.get('FAKE_DOCK_FAIL', '').split(','):
    sys.exit(1)
best = -5 - int(hashlib.sha256(ligand.encode()).hexdigest()[:4], 16) % 50 / 10
shutil.copyfile(args['--ligand'], args['--out'])
print("mode |   affinity | dist from best mode")
print("     | (kcal/mol) | rmsd l.b.| rmsd u.b.")
print("-----+------------+----------+----------")
print(f"   1       {{best:.1f}}      0.000      0.000")
print(f"   2       {{best + 0.5:.1f}}      1.200      2.300")
"""


# The imatinib pocket of Abl (1IEP chain A), the box in fixtures/receptor/config.txt
POCKET_CENTER = (15.6, 53.4, 15.5)
POCKET_RADIUS = 16.0


def write_pocket_receptor(dest):
    """A small receptor cut from the repo's 1IEP.pdb: chain A protein atoms
    within POCKET_RADIUS of the pocket, cleaned the way the wrapper cleans one."""
    with open(REPO / "input-files" / "receptors" / "1IEP.pdb") as pdb, open(dest, "w") as out:
        for line in pdb:
            if not line.startswith("ATOM") or line[21] != "A":
                continue
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            if sum((a - b) ** 2 for a, b in zip(xyz, POCKET_CENTER)) <= POCKET_RADIUS ** 2:
                out.write(line)


def make_collection_tarball(collection_key, dest):
    """<number>/<ligand>.pdbqt, the layout afvs_run.py's unpacker expects."""
    _, collection_number = collection_key.split("_", 1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as tar:
        for ligand_file in sorted((FIXTURES / "ligands" / collection_key).glob("*.pdbqt")):
            tar.add(ligand_file, arcname=f"{collection_number}/{ligand_file.name}")


def local_data_key(collection_key):
    """The S3 key under object_store_data_collection_prefix (metatranche mode)."""
    collection_name, collection_number = collection_key.split("_", 1)
    return f"{collection_name[:2]}/{collection_name}/{collection_number}.tar.gz"


class Screen:
    """One AFVS tree plus its local roots, driven like the wrapper drives it."""

    def __init__(self, base, job_name="fixture-job", workers=1, array_size=1,
                 dockings_per_subjob=3, threads=1, tools_path=None, collections=COLLECTIONS):
        self.base = base
        self.tree = base / "afvs"
        self.tools = self.tree / "tools"
        self.job_root = base / "out" / "afvs"
        self.data_root = base / "collections" / "virtual-screen" / "someone" / "dataset"
        self.tmp = base / "tmp"
        self.job_name = job_name
        self.dock_log = base / "dockings.log"
        self.tools_path = tools_path or (base / "fakebin")

        # Only what the scripts need; the docking binaries stay out of the copy
        shutil.copytree(TOOLS, self.tools, ignore=shutil.ignore_patterns(
            "bin", "docker", "__pycache__", "*.pyc", "cfn"))
        self.tmp.mkdir()

        scenario_dir = self.tree / "input-files" / "qvina02"
        scenario_dir.mkdir(parents=True)
        (self.tree / "input-files" / "receptors").mkdir()
        write_pocket_receptor(self.tree / "input-files" / "receptors" / "receptor.pdb")
        (scenario_dir / "config.txt").write_text(
            (FIXTURES / "receptor" / "config.txt").read_text())

        if tools_path is None:
            self.tools_path.mkdir()
            fake = self.tools_path / "qvina02"
            fake.write_text(FAKE_QVINA.format(python=sys.executable))
            fake.chmod(0o755)

        with open(self.tools / "templates" / "todo.all", "w") as todo:
            for collection_key in collections:
                todo.write(f"{collection_key} {len(ligand_names(collection_key))}\n")
                make_collection_tarball(collection_key, self.data_root / local_data_key(collection_key))

        (self.tools / "templates" / "all.ctrl").write_text(ALL_CTRL.format(
            job_name=job_name, threads=threads, workers=workers, array_size=array_size,
            job_root=self.job_root, data_root=self.data_root, tmp=self.tmp,
            dockings_per_subjob=dockings_per_subjob))

    @property
    def job_dir(self):
        return self.job_root / self.job_name

    def env(self, **extra):
        env = dict(os.environ)
        # The guard makes boto3 unimportable and any inet connect raise
        env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(GUARD_DIR), env.get('PYTHONPATH')]))
        env['AFVS_TOOLS_PATH'] = str(self.tools_path)
        env['FAKE_DOCK_LOG'] = str(self.dock_log)
        env.update(extra)
        return env

    def popen(self, script, *args, **env):
        return subprocess.Popen([sys.executable, f"./{script}", *args], cwd=self.tools,
                                env=self.env(**env), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, start_new_session=True)

    def run(self, script, *args, check=True, timeout=300, **env):
        ret = subprocess.run([sys.executable, f"./{script}", *args], cwd=self.tools,
                             env=self.env(**env), capture_output=True, text=True, timeout=timeout)
        if check and ret.returncode != 0:
            logs = "".join(f"\n--- {p} ---\n{p.read_text()[-4000:]}"
                           for p in sorted((self.tree / "workflow" / "local").rglob("*.out")))
            raise AssertionError(f"{script} {args} exited {ret.returncode}\n{ret.stdout}\n{ret.stderr}{logs}")
        return ret

    def prepare(self):
        self.run("afvs_prepare_folders.py", "--overwrite")
        return self.run("afvs_prepare_workunits.py")

    def status_json(self):
        return json.loads((self.tree / "workflow" / "status.json").read_text())

    def subjobs(self):
        return [(wu, sj) for wu, workunit in self.status_json()['workunits'].items()
                for sj in workunit['subjobs']]

    def done_markers(self):
        return sorted(str(p.relative_to(self.job_dir / ".status"))
                      for p in (self.job_dir / ".status").rglob("*.done"))

    def docking_log(self):
        """[(ligand, start, end)] for every docking the fake engine finished."""
        if not self.dock_log.exists():
            return []
        rows = [line.split() for line in self.dock_log.read_text().splitlines() if line]
        return [(name, float(start), float(end)) for name, start, end in rows]

    def dockings_run(self):
        return sorted(name for name, _, _ in self.docking_log())

    def summary(self, wu, sj):
        with gzip.open(self.job_dir / "summary" / wu / f"{sj}.json.gz", "rt") as f:
            return json.load(f)


@pytest.fixture
def screen(tmp_path):
    return Screen(tmp_path)


@pytest.fixture
def make_screen(tmp_path):
    counter = [0]

    def make(**kwargs):
        counter[0] += 1
        base = tmp_path / f"screen{counter[0]}"
        base.mkdir()
        return Screen(base, **kwargs)

    return make
