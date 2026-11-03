"""The local execution backend: prepare, run, status, resume and recover on one
machine with no AWS Batch and no S3 (batchsystem=local)."""

import fcntl
import gzip
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd
import pytest

from conftest import COLLECTIONS, TOOLS, ligand_names, local_data_key

ALL_LIGANDS = sorted(name for key in COLLECTIONS for name in ligand_names(key))


# The exact extraction the Tamarind wrapper (tamarind-utils
# tool_scripts/run-virtual-screening.sh) runs on afvs_get_status.py's output,
# including check_completion's rule.
WRAPPER_PARSE = r'''
output=$(cat)
submitted=$(echo "$output" | awk '/SUBMITTED/ {print $2}')
runnable=$(echo "$output" | awk '/RUNNABLE/ {print $2}')
pending=$(echo "$output" | awk '/PENDING/ {print $2}')
running=$(echo "$output" | awk '/RUNNING/ {print $2}')
starting=$(echo "$output" | awk '/STARTING/ {print $2}')
succeeded=$(echo "$output" | awk '/SUCCEEDED/ {print $2}')
failed=$(echo "$output" | awk '/FAILED/ {print $2}')
total_succeeded=$(echo "$output" | awk '/- Succeeded/ {print $NF}')
total_failed=$(echo "$output" | awk '/- Failed/ {print $NF}')
total_skipped=$(echo "$output" | awk '/Skipped ligands/ {print $NF}')
vcpu_per_docking=$(echo "$output" | awk '/vCPU seconds per docking/ {print $NF}')
vcpu_total=$(echo "$output" | awk '/vCPU hours total/ {print $NF}')
vcpu_interrupted=$(echo "$output" | awk '/vCPU hours interrupted/ {print $NF}')
complete=0
if [[ "$submitted" == "0" ]] && [[ "$runnable" == "0" ]] && [[ "$pending" == "0" ]] && \
   [[ "$running" == "0" ]] && [[ "$starting" == "0" ]]; then
    complete=1
fi
for v in submitted runnable pending running starting succeeded failed total_succeeded \
         total_failed total_skipped vcpu_per_docking vcpu_total vcpu_interrupted complete; do
    printf '%s=%s\n' "$v" "${!v}"
done
'''


def wrapper_parse(status_output):
    ret = subprocess.run(["bash", "-c", WRAPPER_PARSE], input=status_output,
                         capture_output=True, text=True, check=True)
    return dict(line.split("=", 1) for line in ret.stdout.splitlines())


def status(screen, *args):
    return wrapper_parse(screen.run("afvs_get_status.py", *args).stdout)


def process_group_gone(pgid, timeout=30):
    """No live (non-zombie) member left; unreaped zombies linger in a container."""
    def alive():
        for stat_path in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = stat_path.read_text().rsplit(")", 1)[1].split()
            except (OSError, IndexError):
                continue
            if int(fields[2]) == pgid and fields[0] != "Z":
                return True
        return False

    deadline = time.time() + timeout
    while alive():
        if time.time() >= deadline:
            return False
        time.sleep(0.1)
    return True


def wait_for(predicate, timeout=120, interval=0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("timed out waiting")


# --- prepare --------------------------------------------------------------

def test_prepare_writes_local_workunits(screen):
    out = screen.prepare().stdout
    assert "Generated 2 workunits" in out

    workunits = screen.status_json()['workunits']
    assert sorted(workunits) == ["1", "2"]

    for workunit_id, workunit in workunits.items():
        assert workunit['download_path'] == str(screen.job_dir / "input" / "tasks" / f"{workunit_id}.tar.gz")
        assert set(workunit['subjobs']) == {"0"}
        assert len(workunit['subjobs']['0']['fingerprint']) == 64
        assert workunit['subjobs']['0']['ligands_expected'] == 3

        # The workunit's collections point into local_data_root at the S3 keys
        with tarfile.open(workunit['download_path']) as tar:
            subjob = json.load(tar.extractfile("vf_input/config.json"))['subjobs']['0']
        (collection_key, collection), = subjob['collections'].items()
        assert collection['local_data_path'] == f"{screen.data_root}/{local_data_key(collection_key)}"
        assert 's3_download_path' not in collection

    config = json.loads((screen.tree / "workflow" / "config.json").read_text())
    assert config['local_job_root'] == str(screen.job_root)
    assert config['local_data_root'] == str(screen.data_root)


@pytest.mark.parametrize("old, new, message", [
    # all.ctrl has no inline comments: this parser would keep one as part of the path
    ("local_job_root=", "local_job_root=/out   # comment\nignored_",
     "'local_job_root' must not contain '#'"),
    ("data_storage_mode=local", "data_storage_mode=s3",
     "batchsystem 'local' requires job_storage_mode=local and data_storage_mode=local"),
    # Local storage under AWS Batch stays rejected, as it was before local mode
    ("batchsystem=local", "batchsystem=awsbatch",
     "job_storage_mode=local and data_storage_mode=local need batchsystem=local"),
])
def test_prepare_rejects_mixed_local_configs(screen, old, new, message):
    ctrl = screen.tools / "templates" / "all.ctrl"
    ctrl.write_text(ctrl.read_text().replace(old, new))

    ret = screen.run("afvs_prepare_folders.py", "--overwrite", check=False)
    assert ret.returncode == 1
    assert message in ret.stdout


@pytest.mark.parametrize("edit, error", [
    # The fingerprint must cover the receptor; one it cannot find fails prepare
    (lambda text: text.replace("receptors/receptor.pdb", "receptors/missing.pdb"),
     "receptor = ../input-files/receptors/missing.pdb does not name a file"),
    # Absolute tool locations (dock6_path, chimera_path, ...) are image paths, not inputs
    (lambda text: text + "dock6_path = /opt/dock6\nchimera_path = /usr/bin\n", None),
    # Relative to the folder itself, the way docking resolves it
    (lambda text: text.replace("../input-files/receptors/receptor.pdb", "receptors/receptor.pdb"), None),
    # An absolute receptor (e.g. under /app/inputs) is hashed where it is
    ("absolute", None),
    (lambda text: text.replace("../input-files/receptors/receptor.pdb", "/nonexistent/receptor.pdb"),
     "receptor = /nonexistent/receptor.pdb does not name a file"),
])
def test_scenario_paths_at_prepare(screen, edit, error):
    box = screen.tree / "input-files" / "qvina02" / "config.txt"
    if edit == "absolute":
        receptor = screen.tree / "input-files" / "receptors" / "receptor.pdb"
        box.write_text(box.read_text().replace("../input-files/receptors/receptor.pdb", str(receptor)))
    else:
        box.write_text(edit(box.read_text()))
    screen.run("afvs_prepare_folders.py", "--overwrite")
    ret = screen.run("afvs_prepare_workunits.py", check=False)
    if error:
        assert ret.returncode != 0 and error in ret.stderr
    else:
        assert ret.returncode == 0 and "Generated 2 workunits" in ret.stdout


def test_submit_jobs_points_local_mode_at_the_runner(screen):
    screen.prepare()
    ret = screen.run("afvs_submit_jobs.py", "1", "2", check=False)
    assert ret.returncode == 2
    assert "afvs_run_local.py" in ret.stdout


# --- run + status ---------------------------------------------------------

def test_full_local_screen(screen):
    screen.prepare()

    before = status(screen)
    assert before['pending'] == "2" and before['complete'] == "0"
    assert before['succeeded'] == "0" and before['total_succeeded'] == "0"

    sigmask = screen.base / "sigmask"
    screen.run("afvs_run_local.py", FAKE_DOCK_SIGMASK=str(sigmask))

    # The runner blocks stop signals while it spawns; nothing below inherits that
    masks = sigmask.read_text().split()
    assert len(masks) == 6
    assert all(int(mask, 16) & (1 << (signal.SIGTERM - 1) | 1 << (signal.SIGINT - 1)) == 0 for mask in masks)

    # Same layout as the S3 backend: <scenario>/parquet|logs|csv/<wu>/<sj>.*, summary/<wu>/<sj>.json.gz
    for workunit_id, subjob_id in screen.subjobs():
        assert (screen.job_dir / "qvina02" / "parquet" / workunit_id / f"{subjob_id}.parquet").is_file()
        assert (screen.job_dir / "qvina02" / "logs" / workunit_id / f"{subjob_id}.tar.gz").is_file()
        assert (screen.job_dir / "qvina02" / "csv" / workunit_id / f"{subjob_id}.csv.gz").is_file()
        assert screen.summary(workunit_id, subjob_id)['dockings_status'] == {'success': 3, 'failed': 0}
    assert screen.done_markers() == ["1/0.done", "2/0.done"]
    assert not list(screen.job_dir.rglob("*.partial"))

    scores = pd.concat(pd.read_parquet(p) for p in sorted((screen.job_dir / "qvina02" / "parquet").rglob("*.parquet")))
    assert sorted(scores['ligand']) == ALL_LIGANDS
    assert set(scores['collection_key']) == set(COLLECTIONS)
    assert (scores['score_min'] < 0).all()

    after = status(screen)
    assert after == {
        'submitted': "0", 'runnable': "0", 'pending': "0", 'running': "0", 'starting': "0",
        'succeeded': "2", 'failed': "0",
        'total_succeeded': "6", 'total_failed': "0", 'total_skipped': "0",
        'vcpu_per_docking': after['vcpu_per_docking'], 'vcpu_total': after['vcpu_total'],
        'vcpu_interrupted': "0.00", 'complete': "1",
    }
    assert float(after['vcpu_per_docking']) > 0
    assert screen.dockings_run() == ALL_LIGANDS


def test_status_report_keeps_the_batch_layout(screen):
    screen.prepare()
    screen.run("afvs_run_local.py")
    lines = screen.run("afvs_get_status.py").stdout.splitlines()

    # The table rows are formatted with the AWS Batch report's own format strings
    def row(status_val, workunits, subjobs, ligands):
        return f"{status_val:>15}   {workunits:^10}   {subjobs:^15}   {ligands:^15}"
    assert f"{'Status':>15}   {'Workunits':^10}   {'Subjobs':^15}  {'Dockings (est.)':^15}" in lines
    assert row("SUCCEEDED", 2, 2, 6) in lines
    assert row("FAILED", 0, 0, 0) in lines
    assert row("PENDING", 0, 0, 0) in lines
    assert "  - Succeeded     : 6" in lines
    assert "  - Failed        : 0" in lines
    assert "* Skipped ligands : 0" in lines
    assert "* vCPU hours interrupted  : 0.00" in lines
    assert any(line.startswith("* vCPU hours total        : ") for line in lines)


def test_workers_run_subjobs_in_parallel(make_screen):
    screen = make_screen(workers=2, array_size=2)
    assert "Generated 1 workunits" in screen.prepare().stdout
    assert screen.subjobs() == [("1", "0"), ("1", "1")]

    screen.run("afvs_run_local.py", FAKE_DOCK_SLEEP="1")
    assert screen.done_markers() == ["1/0.done", "1/1.done"]

    # A docking from each collection was in flight at the same moment
    log = screen.docking_log()
    by_collection = [[(start, end) for name, start, end in log if name in ligand_names(key)] for key in COLLECTIONS]
    assert any(a_start < b_end and b_start < a_end
               for a_start, a_end in by_collection[0] for b_start, b_end in by_collection[1])

    assert status(screen)['succeeded'] == "1"


def test_skipped_and_failed_dockings_are_counted(screen):
    screen.prepare()
    failing = ligand_names(COLLECTIONS[0])[0]
    screen.run("afvs_run_local.py", FAKE_DOCK_FAIL=failing)

    parsed = status(screen)
    assert parsed['total_succeeded'] == "5"
    assert parsed['total_failed'] == "1"
    assert parsed['failed'] == "0"  # a failed docking is not a failed subjob
    assert parsed['complete'] == "1"


def test_missing_collection_is_a_failed_download(screen):
    screen.prepare()
    (screen.data_root / local_data_key(COLLECTIONS[1])).unlink()

    screen.run("afvs_run_local.py")

    # As on S3, a missing collection does not fail its subjob
    assert screen.done_markers() == ["1/0.done", "2/0.done"]
    out = screen.run("afvs_get_status.py").stdout
    assert "* Failed Downloads: 1 (est. 3 dockings)" in out
    assert wrapper_parse(out)['total_succeeded'] == "3"
    rows = (screen.tree / "workflow" / "failed_downloads.csv").read_text().splitlines()
    assert len(rows) == 1 and rows[0].startswith(f"2,0,{COLLECTIONS[1]},3,")

    # Rebuilt, not appended, on every call
    screen.run("afvs_get_status.py")
    assert len((screen.tree / "workflow" / "failed_downloads.csv").read_text().splitlines()) == 1


# --- resume ---------------------------------------------------------------

def test_resume_after_sigterm_skips_done_subjobs(screen):
    screen.prepare()
    first, second = screen.subjobs()

    gate, gate_env = screen.gate(COLLECTIONS[1])
    runner = screen.popen("afvs_run_local.py", **gate_env)
    wait_for(lambda: screen.done_markers() == ["1/0.done"])
    running_marker = screen.tree / "workflow" / "local" / "2" / "0.running"
    wait_for(running_marker.exists)
    child_pgid = json.loads(running_marker.read_text())['pid']

    # Mid-run: the wrapper sees it as in progress, and recovery has nothing to do yet
    parsed = status(screen)
    assert parsed['complete'] == "0"
    assert parsed['succeeded'] == "1"
    assert parsed['running'] == "1"
    assert "still running" in screen.run("afvs_recover_failed_subjobs.py", "--queue", "vf-queue-ondemand").stdout

    runner.send_signal(signal.SIGTERM)
    assert runner.wait(timeout=60) == 128 + signal.SIGTERM
    assert screen.done_markers() == ["1/0.done"]
    assert process_group_gone(child_pgid)
    gate.unlink()

    # The interrupted subjob reads as failed until it is rerun
    parsed = status(screen)
    assert parsed['failed'] == "1" and parsed['running'] == "0" and parsed['complete'] == "1"

    done_before = screen.dockings_run()
    first_collection = ligand_names(COLLECTIONS[0])
    assert all(name in done_before for name in first_collection)

    out = screen.run("afvs_run_local.py").stdout
    assert "1 subjobs to run, 1 already done" in out
    assert screen.done_markers() == ["1/0.done", "2/0.done"]

    # The done subjob's ligands were not docked again
    rerun = screen.dockings_run()
    for name in first_collection:
        assert rerun.count(name) == 1
    assert status(screen)['total_succeeded'] == "6"


def test_resume_after_sigkill_of_runner(screen):
    """The runner is killed outright (OOM): no cleanup runs, and its attempt
    keeps going in its own session. The next runner must stop that attempt
    before redoing the subjob."""
    screen.prepare()
    gate, gate_env = screen.gate(COLLECTIONS[1])
    runner = screen.popen("afvs_run_local.py", **gate_env)
    wait_for(lambda: screen.done_markers() == ["1/0.done"])
    running_marker = screen.tree / "workflow" / "local" / "2" / "0.running"
    wait_for(running_marker.exists)

    orphan_pgid = json.loads(running_marker.read_text())['pid']
    runner.kill()
    runner.wait(timeout=30)
    assert not process_group_gone(orphan_pgid, timeout=0)

    parsed = status(screen)
    assert parsed['failed'] == "1" and parsed['running'] == "0"

    gate.unlink()
    out = screen.run("afvs_run_local.py").stdout
    assert "Stopped what was left of an earlier attempt at 2:0" in out
    assert process_group_gone(orphan_pgid, timeout=0)
    assert screen.done_markers() == ["1/0.done", "2/0.done"]
    marker = json.loads((screen.job_dir / ".status" / "2" / "0.done").read_text())
    assert marker['attempts'] == 2
    assert marker['vcpu_seconds_interrupted'] > 0
    assert status(screen)['total_succeeded'] == "6"


def test_recover_after_a_stopped_run_also_runs_what_never_started(make_screen):
    screen = make_screen(array_size=1, collections=COLLECTIONS)
    screen.prepare()
    gate, gate_env = screen.gate(COLLECTIONS[0])
    runner = screen.popen("afvs_run_local.py", **gate_env)
    wait_for((screen.tree / "workflow" / "local" / "1" / "0.running").exists)
    runner.send_signal(signal.SIGTERM)
    assert runner.wait(timeout=60) == 128 + signal.SIGTERM
    gate.unlink()

    # The subjob the stopped run never reached reads FAILED, not PENDING for
    # good, so the wrapper's wait ends and its recovery runs it
    parsed = status(screen)
    assert parsed['failed'] == "2" and parsed['pending'] == "0" and parsed['complete'] == "1"

    screen.run("afvs_recover_failed_subjobs.py", "--queue", "vf-queue-ondemand")
    recovery = status(screen, "--recovery")
    assert recovery['complete'] == "1" and recovery['failed'] == "0"
    assert status(screen)['total_succeeded'] == "6"


def test_resume_across_a_fresh_prepare(screen):
    """A restarted pod re-runs prepare from the same todo.all, with the outputs
    (and their .done markers) restaged under local_job_root."""
    screen.prepare()
    screen.run("afvs_run_local.py")
    assert len(screen.dockings_run()) == 6

    # A restore that also brought back a file the .done does not record
    stray = screen.job_dir / "qvina02" / "csv" / "2" / "0.csv.gz.partial"
    stray.write_text("from a failed attempt")

    screen.prepare()
    out = screen.run("afvs_run_local.py").stdout
    assert "0 subjobs to run, 2 already done" in out
    assert len(screen.dockings_run()) == 6
    assert not stray.exists()

    parsed = status(screen)
    assert parsed['succeeded'] == "2" and parsed['total_succeeded'] == "6"


@pytest.mark.parametrize("change, expect", [
    # Active learning reruns the job under the same name with a new todo.all,
    # so workunit/subjob ids repeat with different contents
    ("todo", "1 subjobs to run, 1 already done"),
    # The same job_name docked against another box, or another receptor
    ("box", "2 subjobs to run, 0 already done"),
    ("receptor", "2 subjobs to run, 0 already done"),
    ("setting", "2 subjobs to run, 0 already done"),
    # A restaged tree that differs only in ways docking ignores still resumes
    ("config_formatting", "0 subjobs to run, 2 already done"),
    ("stray_file", "0 subjobs to run, 2 already done"),
])
def test_done_markers_follow_the_work_they_describe(screen, change, expect):
    screen.prepare()
    screen.run("afvs_run_local.py")

    inputs = screen.tree / "input-files"
    box = inputs / "qvina02" / "config.txt"
    if change == "todo":
        todo = screen.tools / "templates" / "todo.all"
        todo.write_text(todo.read_text().replace(f"{COLLECTIONS[1]} 3", f"{COLLECTIONS[1]} 4"))
    elif change == "box":
        box.write_text(box.read_text().replace("size_x = 18", "size_x = 20"))
    elif change == "receptor":
        receptor = inputs / "receptors" / "receptor.pdb"
        receptor.write_text("".join(receptor.read_text().splitlines(keepends=True)[:-1]))
    elif change == "setting":
        ctrl = screen.tools / "templates" / "all.ctrl"
        ctrl.write_text(ctrl.read_text().replace("run_atom_check=1", "run_atom_check=0"))
    elif change == "config_formatting":
        box.write_text("# restaged\n" + "".join(f"{line.replace(' = ', '=')}   # x\n"
                                                for line in reversed(box.read_text().splitlines())))
    else:
        (inputs / ".DS_Store").write_text("junk")
    screen.prepare()

    out = screen.run("afvs_run_local.py").stdout
    assert expect in out
    rerun = not expect.startswith("0 ")
    assert (".done markers ignored" in out) == rerun
    assert status(screen)['complete'] == "1"


@pytest.mark.parametrize("output", [
    "summary/2/0.json.gz", "qvina02/parquet/2/0.parquet", "qvina02/csv/2/0.csv.gz", "qvina02/logs/2/0.tar.gz"])
@pytest.mark.parametrize("damage", ["delete", "truncate"])
def test_a_done_marker_without_its_outputs_is_not_trusted(screen, output, damage):
    """E.g. a restore from S3 that stopped part way."""
    screen.prepare()
    screen.run("afvs_run_local.py")
    path = screen.job_dir / output
    if damage == "delete":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes()[:-1])

    assert status(screen)['failed'] == "1"
    assert "1 subjobs to run, 1 already done" in screen.run("afvs_run_local.py").stdout


def test_running_means_the_runner_lock_is_held(screen):
    """Liveness is the runner's flock, not a pid that a container can reuse."""
    screen.prepare()
    running_marker = screen.tree / "workflow" / "local" / "1" / "0.running"
    running_marker.parent.mkdir(parents=True)
    # A stale marker naming a live pid (this test's own) must not read as running
    running_marker.write_text(json.dumps({'pid': os.getpid(), 'started': time.time(), 'attempts': 1}))

    assert status(screen)['failed'] == "1"

    lock_path = screen.tree / "workflow" / "local" / "runner.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        parsed = status(screen)
    assert parsed['running'] == "1" and parsed['complete'] == "0"


def test_liveness_is_read_after_the_running_marker(monkeypatch):
    """A status poll racing a runner's start: the runner takes its lock before
    writing any .running marker, so a marker read first and liveness read
    second can never pair a fresh marker with a stale "no runner"."""
    monkeypatch.syspath_prepend(str(TOOLS))
    import afvs_local

    events = []
    monkeypatch.setattr(afvs_local, "read_done_marker", lambda *args: None)
    monkeypatch.setattr(afvs_local, "read_json",
                        lambda path: events.append(("read", path.suffix)) or ({} if path.suffix == ".running" else None))
    monkeypatch.setattr(afvs_local, "runner_alive", lambda: events.append(("alive",)) or True)

    assert afvs_local.subjob_state({}, "1", "0", {'fingerprint': "x"})[0] == afvs_local.RUNNING
    assert events == [("read", ".running"), ("alive",)]

    # The attempt finished (and its runner exited) between the two reads
    done_reads = iter([None, {'fingerprint': "x"}])
    monkeypatch.setattr(afvs_local, "read_done_marker", lambda *args: next(done_reads))
    monkeypatch.setattr(afvs_local, "runner_alive", lambda: False)
    assert afvs_local.subjob_state({}, "1", "0", {'fingerprint': "x"})[0] == afvs_local.SUCCEEDED


def test_kill_orphans_spares_a_reused_pid(tmp_path, monkeypatch):
    """A stale .running whose pid now belongs to another process (its start
    time differs) must not be killed; one that matches is."""
    monkeypatch.syspath_prepend(str(TOOLS))
    import afvs_local
    import afvs_run_local

    monkeypatch.setattr(afvs_local, "LOCAL_STATE_DIR", tmp_path)
    processes = [subprocess.Popen(["sleep", "60"], start_new_session=True) for _ in range(2)]
    try:
        for name, process, recorded in [("0", processes[0], "wrong"), ("1", processes[1], "right")]:
            start_time = afvs_run_local.process_start_time(process.pid)
            (tmp_path / "1").mkdir(exist_ok=True)
            (tmp_path / "1" / f"{name}.running").write_text(json.dumps({
                'pid': process.pid, 'start_time': start_time + 1 if recorded == "wrong" else start_time}))

        afvs_run_local.kill_orphans()
        assert processes[1].wait(timeout=30) == -signal.SIGKILL
        assert processes[0].poll() is None
    finally:
        for process in processes:
            process.kill()
            process.wait()


def test_a_second_runner_is_refused(screen):
    screen.prepare()
    lock_path = screen.tree / "workflow" / "local" / "runner.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        ret = screen.run("afvs_run_local.py", check=False)
    assert ret.returncode == 2
    assert "already running" in ret.stdout
    assert screen.done_markers() == []


# --- failure + recovery ---------------------------------------------------

def test_failed_subjob_exits_nonzero_and_recovers_locally(screen):
    screen.prepare()
    crashing = ligand_names(COLLECTIONS[1])[0]

    ret = screen.run("afvs_run_local.py", check=False, FAKE_DOCK_CRASH=crashing)
    assert ret.returncode == 1
    assert screen.done_markers() == ["1/0.done"]

    parsed = status(screen)
    assert parsed['failed'] == "1" and parsed['complete'] == "1"

    # The wrapper's recovery sequence, unchanged
    screen.run("afvs_recover_failed_subjobs.py", "--queue", "vf-queue-ondemand")
    recovery = status(screen, "--recovery")
    assert recovery['complete'] == "1" and recovery['failed'] == "0"
    assert recovery['total_succeeded'] == ""  # no Completed Summary in the recovery view, as on Batch

    parsed = status(screen)
    assert parsed['succeeded'] == "2" and parsed['failed'] == "0"
    assert parsed['total_succeeded'] == "6"

    marker = json.loads((screen.job_dir / ".status" / "2" / "0.done").read_text())
    assert marker['attempts'] == 2
    assert marker['vcpu_seconds_interrupted'] > 0


def test_workers_of_a_killed_afvs_run_do_not_outlive_it(screen):
    """afvs_run.py OOM-killed while its worker processes live on: they must
    not keep writing outputs into the rerun."""
    screen.prepare()
    victim = ligand_names(COLLECTIONS[0])[0]

    ret = screen.run("afvs_run_local.py", check=False, timeout=100, FAKE_DOCK_KILL_COORDINATOR=victim)
    assert ret.returncode == 1

    failed = json.loads((screen.tree / "workflow" / "local" / "1" / "0.failed").read_text())
    assert failed['returncode'] == -signal.SIGKILL
    # The lingering engine (it sleeps 120 s) and its worker went with it
    assert not any(f"/{victim}.pdbqt" in line for line in ps_lines())


def ps_lines():
    lines = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            lines.append(cmdline.read_bytes().replace(b"\0", b" ").decode())
        except OSError:
            pass
    return lines


@pytest.mark.parametrize("flag", ["--status-file", "--config-file"])
def test_recover_refuses_another_workflows_files(screen, flag):
    screen.prepare()
    other = screen.base / "other.json"
    source = "status.json" if flag == "--status-file" else "config.json"
    other.write_text((screen.tree / "workflow" / source).read_text())
    ret = screen.run("afvs_recover_failed_subjobs.py", flag, str(other), check=False)
    assert ret.returncode == 2 and "not supported with batchsystem=local" in ret.stdout


def test_recover_with_nothing_failed_is_a_no_op(screen):
    screen.prepare()
    screen.run("afvs_run_local.py")
    out = screen.run("afvs_recover_failed_subjobs.py", "--queue", "vf-queue-ondemand").stdout
    assert "No failed subjobs found!" in out
    assert len(screen.dockings_run()) == 6


# --- isolation ------------------------------------------------------------

def test_guard_blocks_boto3_and_the_network(screen):
    """The guard every local-mode test runs under must actually bite, or the
    "no boto3, no network" assertions above are vacuous."""
    probe = (
        "import socket\n"
        "try:\n    import boto3\nexcept ImportError:\n    print('boto3 blocked')\n"
        "try:\n    socket.create_connection(('192.0.2.1', 443), timeout=1)\n"
        "except OSError as err:\n    print('network blocked' if 'test guard' in str(err) else err)\n"
    )
    ret = subprocess.run([sys.executable, "-c", probe], env=screen.env(), capture_output=True, text=True)
    assert ret.stdout.split("\n")[:2] == ["boto3 blocked", "network blocked"]

    # ...and an S3 configuration of the same scripts trips it
    ctrl = screen.tools / "templates" / "all.ctrl"
    ctrl.write_text(ctrl.read_text()
                    .replace("batchsystem=local", "batchsystem=awsbatch\naws_region=us-west-2\naws_batch_array_job_size=10")
                    .replace("job_storage_mode=local", "job_storage_mode=s3\nobject_store_job_bucket=b\nobject_store_job_prefix=p"))
    screen.run("afvs_prepare_folders.py", "--overwrite", "--skip_errors")
    ret = screen.run("afvs_prepare_workunits.py", check=False)
    assert ret.returncode != 0 and "boto3" in ret.stderr
