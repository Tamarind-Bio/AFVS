"""Shared test fixtures.

The fixture library is two collections of three ligands each (the file names
under fixtures/ligands/<collection_key>/), a fake docking engine that prints
qvina02-style output, and a small receptor cut from the repo's 1IEP.pdb for
the real-engine test. Presigned-mode jobs run as subprocesses under
guard/sitecustomize.py (boto3 unimportable, only loopback reachable).
"""

import os
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# AFVS_TOOLS=<dir> runs the suite against another copy of tools/ (e.g. develop's)
TOOLS = Path(os.environ.get("AFVS_TOOLS", REPO / "tools"))
FIXTURES = Path(__file__).resolve().parent / "fixtures"
GUARD_DIR = Path(__file__).resolve().parent / "guard"

COLLECTIONS = ["AACAAA_00001", "AACAAB_00002"]


def ligand_names(collection_key):
    return sorted(p.stem for p in (FIXTURES / "ligands" / collection_key).glob("*.pdbqt"))


FAKE_QVINA = """\
#!{python}
# Fake qvina02: prints a vina-style result table for the ligand.
#   FAKE_DOCK_SLEEP     seconds to sleep per docking
#   FAKE_DOCK_FAIL      comma-separated ligand names that exit 1
#   FAKE_DOCK_CRASH     ligand name that kills its whole afvs_run.py (a hard subjob failure)
#   FAKE_DOCK_KILL_COORDINATOR  ligand name that kills only afvs_run.py itself
#   FAKE_DOCK_GATE      file; while it exists, FAKE_DOCK_GATE_LIGANDS (comma-separated) wait
#   FAKE_DOCK_SIGMASK   file that gets the engine's blocked-signal mask per docking
#   FAKE_DOCK_LOG       file that gets "<ligand> <start> <end>" per docking run
import hashlib, os, shutil, signal, sys, time
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
ligand = os.path.basename(args['--ligand']).split('.')[0]
start = time.time()
time.sleep(float(os.environ.get('FAKE_DOCK_SLEEP', '0')))
if ligand in os.environ.get('FAKE_DOCK_GATE_LIGANDS', '').split(','):
    while os.path.exists(os.environ['FAKE_DOCK_GATE']):
        time.sleep(0.05)
if ligand == os.environ.get('FAKE_DOCK_CRASH'):
    os.killpg(os.getpgrp(), signal.SIGKILL)
if ligand == os.environ.get('FAKE_DOCK_KILL_COORDINATOR'):
    # Kill only afvs_run.py (our grandparent), as an OOM kill would, and
    # linger the way its surviving worker processes do
    with open(f"/proc/{{os.getppid()}}/stat") as stat:
        coordinator = int(stat.read().rsplit(")", 1)[1].split()[1])
    os.kill(coordinator, signal.SIGKILL)
    time.sleep(120)
if os.environ.get('FAKE_DOCK_SIGMASK'):
    with open('/proc/self/status') as status, open(os.environ['FAKE_DOCK_SIGMASK'], 'a') as out:
        out.write(next(line for line in status if line.startswith('SigBlk:')).split()[1] + "\\n")
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


def data_key(collection_key):
    """The S3 key under object_store_data_collection_prefix (metatranche mode)."""
    collection_name, collection_number = collection_key.split("_", 1)
    return f"{collection_name[:2]}/{collection_name}/{collection_number}.tar.gz"
