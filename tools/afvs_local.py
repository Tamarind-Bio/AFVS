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
# Description: Shared state for the local execution backend
#              (batchsystem=local, job_storage_mode=local, data_storage_mode=local)
#
# A local screen runs every subjob on this machine. Its outputs land under
# <local_job_root>/<job_name>/ with the same layout the S3 backend writes under
# <object_store_job_prefix>/<job_name>/. A subjob is complete once
# <local_job_root>/<job_name>/.status/<wu>/<sj>.done exists; it is written only
# after that subjob's outputs are complete, and a rerun skips subjobs that have it.
#
# Per-attempt bookkeeping (running / failed markers, logs) lives under
# ../workflow/local/, which afvs_prepare_folders.py resets for every wave.
#
# Nothing in this module (or anything a local-mode path imports) uses boto3 or
# the network.
#
# ---------------------------------------------------------------------------

import gzip
import hashlib
import json
import os
import socket
from pathlib import Path


WORKFLOW_DIR = Path("..") / "workflow"
LOCAL_STATE_DIR = WORKFLOW_DIR / "local"
RUNNER_INFO = LOCAL_STATE_DIR / "runner.json"
RUNNER_LOCK = LOCAL_STATE_DIR / "runner.lock"

# Subjob states, named after the AWS Batch states afvs_get_status.py reports
SUCCEEDED = "SUCCEEDED"
RUNNING = "RUNNING"
FAILED = "FAILED"
PENDING = "PENDING"

# The parts of a collection entry that decide what a subjob docks. Paths are
# left out so a screen restaged under a different mount still matches.
_FINGERPRINT_COLLECTION_KEYS = ('count', 'mode', 'ligands', 'prescreen_ligands_per_tranche', 'dockings')


def is_local(config):
    return config.get('batchsystem') == "local"


def job_root(config):
    return Path(config['local_job_root']) / config['job_name']


def done_marker_path(config, workunit_id, subjob_id):
    return job_root(config) / ".status" / str(workunit_id) / f"{subjob_id}.done"


def attempt_path(workunit_id, subjob_id, kind):
    return LOCAL_STATE_DIR / str(workunit_id) / f"{subjob_id}.{kind}"


def summary_path(config, workunit_id, subjob_id):
    return job_root(config) / "summary" / str(workunit_id) / f"{subjob_id}.json.gz"


def subjob_output_paths(config, workunit_id, subjob_id):
    """Every file afvs_run.py can write for one subjob (see generate_output_path)."""
    root = job_root(config)
    paths = [summary_path(config, workunit_id, subjob_id)]
    for scenario_key in config['docking_scenario_names']:
        paths.append(root / scenario_key / "parquet" / str(workunit_id) / f"{subjob_id}.parquet")
        paths.append(root / scenario_key / "csv" / str(workunit_id) / f"{subjob_id}.csv.gz")
        paths.append(root / scenario_key / "logs" / str(workunit_id) / f"{subjob_id}.tar.gz")
    return paths


def subjob_fingerprint(config, subjob):
    """Identifies what a subjob docks, so a .done left by an earlier wave that
    reused the same workunit/subjob ids is not mistaken for this wave's."""
    collections = {}
    for collection_key, collection in subjob['collections'].items():
        collections[collection_key] = {
            key: collection[key] for key in _FINGERPRINT_COLLECTION_KEYS if key in collection
        }

    blob = json.dumps({
        'collections': collections,
        'scenario_names': config['docking_scenario_names'],
        'scenario_methods': config['docking_scenario_methods'],
        'scenario_replicas': config['docking_scenario_replicas'],
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def read_json(path):
    try:
        with open(path, "r") as read_file:
            return json.load(read_file)
    except (FileNotFoundError, ValueError):
        return None


def write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "w") as json_out:
        json.dump(data, json_out)
        json_out.flush()
        os.fsync(json_out.fileno())
    os.replace(tmp_path, path)


def remove_file(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def read_summary(config, workunit_id, subjob_id):
    try:
        with gzip.open(summary_path(config, workunit_id, subjob_id), "rt") as read_file:
            return json.load(read_file)
    except (FileNotFoundError, OSError, EOFError, ValueError):
        return None


def read_done_marker(config, workunit_id, subjob_id, fingerprint):
    """The subjob's .done contents if it is complete for THIS workunit layout,
    otherwise None (missing, unreadable, another wave's, or its summary is gone)."""
    marker = read_json(done_marker_path(config, workunit_id, subjob_id))
    if not isinstance(marker, dict) or marker.get('fingerprint') != fingerprint:
        return None
    if not summary_path(config, workunit_id, subjob_id).is_file():
        return None
    return marker


def runner_alive():
    """True when an afvs_run_local.py on this host still owns the workflow."""
    info = read_json(RUNNER_INFO)
    if not isinstance(info, dict) or info.get('host') != socket.gethostname():
        return False
    try:
        os.kill(int(info['pid']), 0)
    except (KeyError, TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def subjob_state(config, workunit_id, subjob_id, subjob, alive=None):
    """(state, marker) for one subjob.

    SUCCEEDED  valid .done for this layout
    RUNNING    an attempt is in flight under a live runner
    FAILED     the last attempt failed, or its runner died mid-attempt
    PENDING    never attempted in this wave
    """
    marker = read_done_marker(config, workunit_id, subjob_id, subjob.get('fingerprint'))
    if marker is not None:
        return SUCCEEDED, marker

    running = read_json(attempt_path(workunit_id, subjob_id, "running"))
    if running is not None:
        if alive is None:
            alive = runner_alive()
        if alive:
            return RUNNING, running
        return FAILED, running

    failed = read_json(attempt_path(workunit_id, subjob_id, "failed"))
    if failed is not None:
        return FAILED, failed

    return PENDING, None


def workunit_state(subjob_states):
    """Roll subjob states up the way an AWS Batch array parent reports them."""
    if all(state == SUCCEEDED for state in subjob_states):
        return SUCCEEDED
    if any(state == RUNNING for state in subjob_states):
        return RUNNING
    if any(state == PENDING for state in subjob_states):
        return PENDING
    return FAILED
