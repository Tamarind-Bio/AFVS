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
import socket
import subprocess
import sys
import time
from pathlib import Path

import afvs_local


TOOLS_DIR = Path(__file__).resolve().parent
AFVS_RUN = TOOLS_DIR / "templates" / "afvs_run.py"

POLL_SECONDS = 0.2


class Terminated(Exception):
    pass


def _raise_terminated(signum, frame):
    raise Terminated(signum)


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


def outputs_complete(config, workunit_id, subjob_id):
    """afvs_run.py exited 0; check its outputs are actually in place before the
    subjob is marked done. The summary is always written; the per-scenario log
    archives are written whenever anything was docked."""
    summary = afvs_local.read_summary(config, workunit_id, subjob_id)
    if summary is None:
        return False

    if summary.get('total_dockings', 0) > 0:
        root = afvs_local.job_root(config)
        for scenario_key in config['docking_scenario_names']:
            if not (root / scenario_key / "logs" / str(workunit_id) / f"{subjob_id}.tar.gz").is_file():
                return False

    return True


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
        previous = afvs_local.read_json(self.failed_path) or afvs_local.read_json(self.running_path) or {}
        self.attempts = int(previous.get('attempts', 0)) + 1
        self.vcpu_seconds_interrupted = float(previous.get('vcpu_seconds_interrupted', 0))

        self.process = None
        self.started = None

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
                start_new_session=True
            )

        afvs_local.write_json_atomic(self.running_path, {
            'pid': self.process.pid,
            'started': self.started,
            'attempts': self.attempts,
            'vcpu_seconds_interrupted': self.vcpu_seconds_interrupted,
        })

    def vcpu_seconds(self):
        return (time.time() - self.started) * int(self.config['threads_to_use'])

    def finish(self):
        """Record the result of an exited attempt. Returns True on success."""
        returncode = self.process.returncode
        vcpu_seconds = self.vcpu_seconds()

        if returncode == 0 and outputs_complete(self.config, self.workunit_id, self.subjob_id):
            afvs_local.write_json_atomic(
                afvs_local.done_marker_path(self.config, self.workunit_id, self.subjob_id), {
                    'fingerprint': self.subjob['fingerprint'],
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
        if self.process is None:
            afvs_local.remove_file(self.running_path)
            return
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait()
        self.mark_failed(self.process.returncode, self.vcpu_seconds())


def acquire_runner_lock():
    afvs_local.LOCAL_STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = open(afvs_local.RUNNER_LOCK, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return None
    return lock_file


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

            for attempt in [attempt for attempt in running if attempt.process.poll() is not None]:
                running.remove(attempt)
                if not attempt.finish():
                    failed += 1

            time.sleep(POLL_SECONDS)
    finally:
        # Only reached with attempts still running if we were stopped
        if running:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        for attempt in running:
            attempt.stop()

    return failed


def run(config, workers, select=None):
    """Run every subjob that is not done (or, with `select`, those of the given
    states). Returns a process exit status."""

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

    try:
        afvs_local.write_json_atomic(afvs_local.RUNNER_INFO, {
            'pid': os.getpid(),
            'host': socket.gethostname(),
            'started': time.time()
        })

        workunits = get_workunits()

        todo = []
        skipped = 0
        for workunit_id, workunit in workunits.items():
            for subjob_id, subjob in workunit['subjobs'].items():
                state, _ = afvs_local.subjob_state(config, workunit_id, subjob_id, subjob, alive=False)
                if state == afvs_local.SUCCEEDED:
                    skipped += 1
                elif select is None or state in select:
                    todo.append((workunit_id, subjob_id))

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
        afvs_local.remove_file(afvs_local.RUNNER_INFO)
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
