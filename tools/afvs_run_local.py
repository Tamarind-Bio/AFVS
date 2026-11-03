#!/usr/bin/env python3

# Copyright (C) 2026 Tamarind Bio
#
# This file is part of AdaptiveFlow.
#
# AdaptiveFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# AdaptiveFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with AdaptiveFlow.  If not, see <https://www.gnu.org/licenses/>.

# ---------------------------------------------------------------------------
#
# Description: Run every pending subjob of a batchsystem=local workflow on
#              this machine, blocking until they have all finished.
#
#   ./afvs_prepare_folders.py --overwrite
#   ./afvs_prepare_workunits.py
#   ./afvs_run_local.py [--workers N]
#
# Each subjob runs templates/afvs_run.py exactly as an AWS Batch array child
# would, with job_storage_mode=local. Subjobs whose .done marker is already in
# place (a resumed screen) are skipped. Exits nonzero if any subjob it ran
# still failed.
#
# ---------------------------------------------------------------------------

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import afvs_local


TOOLS_DIR = Path(__file__).resolve().parent
AFVS_RUN = TOOLS_DIR / "templates" / "afvs_run.py"

POLL_SECONDS = 0.2
LOCK_WAIT_SECONDS = 5
ORPHAN_WAIT_SECONDS = 30
STOP_SIGNALS = {signal.SIGTERM, signal.SIGINT}


class Terminated(Exception):
    pass


def _raise_terminated(signum, frame):
    # One stop is enough: ignore any further signal so cleanup runs to the end
    for stop_signal in STOP_SIGNALS:
        signal.signal(stop_signal, signal.SIG_IGN)
    raise Terminated(signum)


def _unblock_stop_signals():
    signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)


def kill_process_group(pgid):
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def process_start_time(pid):
    """The kernel's start time of a process (clock ticks since boot), or None.
    With the pid it identifies one process: a reused pid has another."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def process_group_alive(pgid):
    """Whether any process of the group is still running. Killed orphans can
    linger as zombies when nothing reaps them (a container's PID 1 often
    does not), so on Linux zombies are not counted."""
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False

    proc = Path("/proc")
    if not (proc / "self" / "stat").exists():
        return True
    for stat_path in proc.glob("[0-9]*/stat"):
        try:
            # pid (comm) state ppid pgrp ...; comm may contain spaces
            fields = stat_path.read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if int(fields[2]) == pgid and fields[0] != "Z":
            return True
    return False


def wait_for_process_group(pgid, timeout):
    deadline = time.time() + timeout
    while process_group_alive(pgid):
        if time.time() >= deadline:
            return False
        time.sleep(POLL_SECONDS)
    return True


def kill_orphans():
    """An attempt whose runner was killed outright (SIGKILL, OOM) keeps running
    in its own session. Stop each one before its subjob is run again, so two
    attempts never write the same outputs. Called with the runner lock held,
    so every .running marker left is a dead runner's."""
    for running_path in sorted(afvs_local.LOCAL_STATE_DIR.glob("*/*.running")):
        marker = afvs_local.read_json(running_path) or {}
        pgid = marker.get('pid')
        # A reused pid could name our own group; never kill that
        if not isinstance(pgid, int) or pgid <= 1 or pgid in (os.getpid(), os.getpgrp()):
            continue
        # If the group's leader is still there it must be the process the
        # marker recorded. (If it is gone, the kernel does not hand its pid out
        # as a new group id while members of the old group live, so any live
        # member is the orphan's.)
        leader_start = process_start_time(pgid)
        if leader_start is not None and marker.get('start_time') not in (None, leader_start):
            continue
        if process_group_alive(pgid):
            # It ran until now: record that, so its time counts as interrupted
            kill_process_group(pgid)
            marker['stopped'] = time.time()
            afvs_local.write_json_atomic(running_path, marker)
        if wait_for_process_group(pgid, ORPHAN_WAIT_SECONDS):
            print(f"Stopped what was left of an earlier attempt at {running_path.parent.name}:{running_path.stem}", flush=True)
        else:
            print(f"WARNING: process group {pgid} of an earlier attempt did not exit", flush=True)


def parse_config(filename):
    with open(filename, "r") as read_file:
        config = json.load(read_file)

    return config


def get_workunits():
    with open(afvs_local.WORKFLOW_DIR / "status.json", "r") as read_file:
        return json.load(read_file)['workunits']


def subjob_env(config, workunit_id, subjob_id, workunit):
    env = dict(os.environ)
    env.update({
        'AFVS_RUN_MODE': "local",
        'AFVS_JOB_STORAGE_MODE': "local",
        'AFVS_WORKUNIT': str(workunit_id),
        'AFVS_WORKUNIT_SUBJOB': str(subjob_id),
        'AFVS_CONFIG_JOB_TGZ': workunit['download_path'],
        'AFVS_VCPUS': str(config['threads_to_use']),
        'AFVS_TMP_PATH': config['tempdir_default'],
        'AFVS_RUN_SEQUENTIAL': "0",
    })
    env.setdefault('AFVS_TOOLS_PATH', str(TOOLS_DIR / "bin"))
    return env


class Attempt:

    def __init__(self, config, workunit_id, subjob_id, workunit):
        self.config = config
        self.workunit_id = workunit_id
        self.subjob_id = subjob_id
        self.subjob = workunit['subjobs'][subjob_id]
        self.workunit = workunit

        self.running_path = afvs_local.attempt_path(workunit_id, subjob_id, "running")
        self.failed_path = afvs_local.attempt_path(workunit_id, subjob_id, "failed")
        self.log_path = afvs_local.attempt_path(workunit_id, subjob_id, "out")

        # A failed attempt, or one whose runner died mid-attempt
        previous = afvs_local.read_json(self.failed_path)
        if previous is None:
            previous = afvs_local.read_json(self.running_path) or {}
            if 'started' in previous:
                # Its runner died, so nobody recorded its time: count it up to
                # when kill_orphans stopped it, else to the last line it logged
                try:
                    ended = previous.get('stopped') or os.path.getmtime(self.log_path)
                    previous['vcpu_seconds_interrupted'] = float(previous.get('vcpu_seconds_interrupted', 0)) + \
                        max(0.0, ended - float(previous['started'])) * int(config['threads_to_use'])
                except (OSError, TypeError, ValueError):
                    pass
        self.attempts = int(previous.get('attempts', 0)) + 1
        self.vcpu_seconds_interrupted = float(previous.get('vcpu_seconds_interrupted', 0))

        self.process = None
        self.started = None
        self.finished = False

    def start(self):
        # Clear what an earlier attempt (or an earlier wave) left, so nothing
        # stale survives into this attempt's outputs
        afvs_local.remove_file(afvs_local.done_marker_path(self.config, self.workunit_id, self.subjob_id))
        for output_path in afvs_local.subjob_output_paths(self.config, self.workunit_id, self.subjob_id):
            afvs_local.remove_file(output_path)
        afvs_local.remove_file(self.failed_path)

        self.started = time.time()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Workunit {self.workunit_id}:{self.subjob_id}: started (attempt {self.attempts}), output in {self.log_path}", flush=True)

        # A stop signal is held back until the child is recorded (self.process
        # and its .running marker), so stop() can always find it
        try:
            signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
            self._spawn()
        finally:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)

    def _spawn(self):
        with open(self.log_path, "w") as log_out:
            # A session of its own, so afvs_run.py's worker processes can be
            # stopped together with it
            self.process = subprocess.Popen(
                [sys.executable, str(AFVS_RUN)],
                cwd=TOOLS_DIR,
                env=subjob_env(self.config, self.workunit_id, self.subjob_id, self.workunit),
                stdin=subprocess.DEVNULL,
                stdout=log_out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                # The child inherits the mask start() set; give it the default
                preexec_fn=_unblock_stop_signals
            )

        afvs_local.write_json_atomic(self.running_path, {
            'pid': self.process.pid,
            'start_time': process_start_time(self.process.pid),
            'started': self.started,
            'attempts': self.attempts,
            'vcpu_seconds_interrupted': self.vcpu_seconds_interrupted,
        })

    def vcpu_seconds(self):
        return (time.time() - self.started) * int(self.config['threads_to_use'])

    def exited(self):
        """Whether afvs_run.py has exited, WITHOUT reaping it: until it is
        reaped its pid (= its process group id) cannot be reused, so finish()
        can safely kill the group."""
        try:
            return os.waitid(os.P_PID, self.process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
        except ChildProcessError:
            return self.process.poll() is not None

    def finish(self):
        """Record the result of an exited attempt. Returns True on success."""
        # afvs_run.py's worker processes share its session; if it died without
        # them (OOM), they must not outlive it and race a rerun
        kill_process_group(self.process.pid)
        returncode = self.process.wait()
        self.finished = True
        vcpu_seconds = self.vcpu_seconds()

        if returncode == 0 and afvs_local.outputs_complete(self.config, self.workunit_id, self.subjob_id):
            afvs_local.write_json_atomic(
                afvs_local.done_marker_path(self.config, self.workunit_id, self.subjob_id), {
                    'fingerprint': self.subjob['fingerprint'],
                    'outputs': afvs_local.output_manifest(self.config, self.workunit_id, self.subjob_id),
                    'vcpu_seconds': vcpu_seconds,
                    'vcpu_seconds_interrupted': self.vcpu_seconds_interrupted,
                    'attempts': self.attempts,
                    'started': self.started,
                    'finished': time.time(),
                })
            afvs_local.remove_file(self.running_path)
            print(f"Workunit {self.workunit_id}:{self.subjob_id}: succeeded", flush=True)
            return True

        self.mark_failed(returncode, vcpu_seconds)
        print(f"Workunit {self.workunit_id}:{self.subjob_id}: FAILED (exit {returncode}), see {self.log_path}", flush=True)
        return False

    def mark_failed(self, returncode, vcpu_seconds):
        afvs_local.write_json_atomic(self.failed_path, {
            'returncode': returncode,
            'attempts': self.attempts,
            'vcpu_seconds_interrupted': self.vcpu_seconds_interrupted + vcpu_seconds,
        })
        afvs_local.remove_file(self.running_path)

    def stop(self):
        if self.finished:
            # finish() already recorded it; its pid may be reused by now
            return
        if self.process is None:
            afvs_local.remove_file(self.running_path)
            return
        kill_process_group(self.process.pid)
        self.process.wait()
        self.mark_failed(self.process.returncode, self.vcpu_seconds())


def acquire_runner_lock():
    """The runner lock, or None if another runner holds it. A status call
    probes the lock (shared, for an instant), so retry for a few seconds
    before giving up."""
    afvs_local.LOCAL_STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = open(afvs_local.RUNNER_LOCK, "w")
    deadline = time.time() + LOCK_WAIT_SECONDS
    while True:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_file
        except BlockingIOError:
            if time.time() >= deadline:
                lock_file.close()
                return None
            time.sleep(POLL_SECONDS)


def run_subjobs(config, workunits, subjobs, workers):
    """Run (workunit_id, subjob_id) pairs, at most `workers` at a time.
    Returns the number that failed."""

    queue = list(subjobs)
    running = []
    failed = 0

    try:
        while queue or running:
            while queue and len(running) < workers:
                workunit_id, subjob_id = queue.pop(0)
                attempt = Attempt(config, workunit_id, subjob_id, workunits[workunit_id])
                running.append(attempt)
                attempt.start()

            for attempt in [attempt for attempt in running if attempt.exited()]:
                if not attempt.finish():
                    failed += 1
                running.remove(attempt)

            time.sleep(POLL_SECONDS)
    finally:
        # Only reached with attempts still running if we were stopped (the
        # signal handler has already set further stop signals to be ignored)
        for attempt in running:
            attempt.stop()

    return failed


def run(config, workers):
    """Run every subjob that is not done. Returns a process exit status."""

    if not afvs_local.is_local(config):
        print(f"afvs_run_local.py needs batchsystem=local (this workflow has batchsystem={config['batchsystem']})")
        return 2

    lock_file = acquire_runner_lock()
    if lock_file is None:
        print("Another afvs_run_local.py is already running this workflow")
        return 2

    previous_handlers = {
        signum: signal.signal(signum, _raise_terminated) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    # An inherited SIG_IGN would have the kernel reap subjobs itself, and a
    # failed afvs_run.py would then read as exit 0
    previous_handlers[signal.SIGCHLD] = signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    try:
        kill_orphans()
        afvs_local.RUN_STARTED.write_text(f"{time.time()}\n")

        workunits = get_workunits()

        todo = []
        skipped = 0
        ignored_markers = 0
        for workunit_id, workunit in workunits.items():
            for subjob_id, subjob in workunit['subjobs'].items():
                state, marker = afvs_local.subjob_state(config, workunit_id, subjob_id, subjob, alive=False)
                if state == afvs_local.SUCCEEDED:
                    afvs_local.remove_unrecorded_outputs(config, workunit_id, subjob_id, marker)
                    skipped += 1
                else:
                    todo.append((workunit_id, subjob_id))
                    if afvs_local.done_marker_path(config, workunit_id, subjob_id).exists():
                        ignored_markers += 1

        if ignored_markers:
            print(f"{ignored_markers} .done markers ignored: they are for different work "
                  "(another todo.all, receptor or box) or their outputs are incomplete", flush=True)
        print(f"{len(todo)} subjobs to run, {skipped} already done, {workers} at a time", flush=True)

        failed = run_subjobs(config, workunits, todo, workers)

        print(f"{len(todo) - failed} subjobs succeeded, {failed} failed", flush=True)
        return 1 if failed else 0

    except Terminated as stop:
        print(f"Stopped by signal {stop.args[0]}", flush=True)
        return 128 + stop.args[0]

    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        lock_file.close()


def main():
    parser = argparse.ArgumentParser(description="Run a batchsystem=local workflow on this machine")
    parser.add_argument('--workers', type=int,
        help="Subjobs to run at once (default: local_workers from all.ctrl, else 1)")
    args = parser.parse_args()

    config = parse_config(afvs_local.WORKFLOW_DIR / "config.json")

    workers = args.workers if args.workers is not None else int(config.get('local_workers') or 1)
    if workers < 1:
        parser.error("--workers must be at least 1")

    return run(config, workers)


if __name__ == '__main__':
    sys.exit(main())
