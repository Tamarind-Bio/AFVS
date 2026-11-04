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
# Per-attempt bookkeeping (running / failed markers, logs)
# lives under ../workflow/local/, which afvs_prepare_folders.py resets for
# every wave.
#
# Nothing in this module (or anything a local-mode path imports) uses boto3 or
# the network.
#
# ---------------------------------------------------------------------------

import errno
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path


WORKFLOW_DIR = Path("..") / "workflow"
LOCAL_STATE_DIR = WORKFLOW_DIR / "local"
# Held (flock LOCK_EX) by afvs_run_local.py for as long as it owns the
# workflow. It sits beside ../workflow, not in it: prepare --overwrite deletes
# that directory while holding the lock, and a lock file deleted with it would
# let a second runner lock a fresh file at the same path.
RUNNER_LOCK = Path("..") / ".afvs_local_runner.lock"
# Written when a runner starts, so status can tell "not run yet" from "a run
# ended without reaching this subjob"
RUN_STARTED = LOCAL_STATE_DIR / "started"

# Subjob states, named after the AWS Batch states afvs_get_status.py reports
SUCCEEDED = "SUCCEEDED"
RUNNING = "RUNNING"
FAILED = "FAILED"
PENDING = "PENDING"

# The parts of a collection entry that decide what a subjob docks. Paths are
# left out so a screen restaged under a different mount still matches.
_FINGERPRINT_COLLECTION_KEYS = ('count', 'mode', 'ligands', 'prescreen_ligands_per_tranche', 'dockings')

# all.ctrl settings that change which ligands are docked or what is written
_FINGERPRINT_CONFIG_KEYS = (
    'docking_scenario_names', 'docking_scenario_methods', 'docking_scenario_replicas',
    'ligand_library_format', 'run_atom_check', 'prescreen_mode', 'prescreen_ligands_per_tranche',
    'dynamic_tranche_filtering', 'dynamic_tranche_filtering_regex', 'summary_formats',
    'print_attrs_in_summary', 'threads_per_docking', 'program_timeout',
)


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


def scenario_output_path(config, scenario_key, content_type, workunit_id, subjob_id):
    extension = {'parquet': "parquet", 'csv': "csv.gz", 'logs': "tar.gz"}[content_type]
    return job_root(config) / scenario_key / content_type / str(workunit_id) / f"{subjob_id}.{extension}"


def output_scenarios(config):
    """The configured scenarios, plus any other scenario folder already in the
    job root (left by an earlier run with other scenarios)."""
    scenarios = list(config['docking_scenario_names'])
    root = job_root(config)
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if path.is_dir() and path.name not in scenarios and path.name not in ("summary", "input", ".status") \
                    and any((path / content_type).is_dir() for content_type in ('parquet', 'csv', 'logs')):
                scenarios.append(path.name)
    return scenarios


def subjob_output_paths(config, workunit_id, subjob_id):
    """Every file afvs_run.py can write for one subjob (see generate_output_path),
    under the configured scenarios and any left from an earlier run, including
    the .partial name each is written under before it is renamed."""
    paths = [summary_path(config, workunit_id, subjob_id)]
    for scenario_key in output_scenarios(config):
        for content_type in ('parquet', 'csv', 'logs'):
            paths.append(scenario_output_path(config, scenario_key, content_type, workunit_id, subjob_id))
    return paths + [path.with_name(f"{path.name}.partial") for path in paths]


def outputs_complete(config, workunit_id, subjob_id):
    """Whether every output afvs_run.py writes for this subjob is in place.

    The summary is always written. Whenever anything was docked, each scenario
    gets a log archive. The score tables (parquet / csv.gz, per summary_formats)
    exist when a docking succeeded; with several scenarios afvs_run.py only
    writes them up to the first scenario that had no success, so they are
    required here for the single-scenario case only."""
    summary = read_summary(config, workunit_id, subjob_id)
    if summary is None:
        return False

    if summary.get('total_dockings', 0) > 0:
        for scenario_key in config['docking_scenario_names']:
            if not scenario_output_path(config, scenario_key, 'logs', workunit_id, subjob_id).is_file():
                return False

    if summary.get('dockings_status', {}).get('success', 0) > 0 and len(config['docking_scenario_names']) == 1:
        scenario_key = config['docking_scenario_names'][0]
        summary_formats = config.get('summary_formats') or []
        for summary_format, content_type in (('parquet', 'parquet'), ('csv.gz', 'csv')):
            if summary_format in summary_formats and \
                    not scenario_output_path(config, scenario_key, content_type, workunit_id, subjob_id).is_file():
                return False

    return True


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as read_file:
        for chunk in iter(lambda: read_file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inputs_digest(config):
    """Hash of everything docking reads from docking_scenario_basefolder:
    every file under it (each scenario's config.txt, the receptor, grids,
    rDock .prm files and whatever those refer to), byte for byte. Hidden
    files are left out, so a stray .DS_Store in a restaged tree does not
    invalidate every .done (a config that names a hidden file is therefore
    not covered). Symlinked directories are followed, as the copy of this
    folder into each workunit follows them."""
    base = Path(config['docking_scenario_basefolder'])
    files = {}
    for directory, dirnames, filenames in os.walk(base, followlinks=True):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        for filename in sorted(filenames):
            path = Path(directory) / filename
            if not filename.startswith(".") and path.is_file():
                files[path.relative_to(base).as_posix()] = _file_hash(path)

    # Absolute files a scenario config names (e.g. receptor = /app/inputs/r.pdb)
    for scenario_key in config['docking_scenario_names']:
        config_path = base / scenario_key / "config.txt"
        if not config_path.is_file():
            continue
        for line in config_path.read_text().splitlines():
            for token in line.split('#')[0].replace("=", " ").split():
                token = token.strip('"\',;')
                if os.path.isabs(token) and os.path.isfile(token):
                    files[token] = _file_hash(Path(token))
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def collection_sizes(subjob):
    """Byte size of each staged collection tarball (None if missing). A cheap
    guard against a restage that changed a collection under the same key;
    hashing contents would read the whole library on every prepare."""
    sizes = {}
    for collection_key, collection in subjob['collections'].items():
        try:
            sizes[collection_key] = os.path.getsize(collection['local_data_path'])
        except (KeyError, OSError):
            sizes[collection_key] = None
    return sizes


def subjob_fingerprint(config, subjob, docking_inputs):
    """Identifies what a subjob docks, so a .done left by an earlier wave that
    reused the same workunit/subjob ids (or an earlier run under the same
    job_name with another receptor or box) is not mistaken for this one's.
    `docking_inputs` is inputs_digest(config)."""
    collections = {}
    for collection_key, collection in subjob['collections'].items():
        collections[collection_key] = {
            key: collection[key] for key in _FINGERPRINT_COLLECTION_KEYS if key in collection
        }

    blob = json.dumps({
        'collections': collections,
        'collection_sizes': collection_sizes(subjob),
        'config': {key: config.get(key) for key in _FINGERPRINT_CONFIG_KEYS},
        'docking_inputs': docking_inputs,
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


def output_manifest(config, workunit_id, subjob_id):
    """{path relative to the job root: size} of the subjob's outputs on disk.
    Recorded in its .done, so a resume can tell a partial restore (any
    scenario, any format) from a complete one."""
    root = job_root(config)
    manifest = {}
    for path in subjob_output_paths(config, workunit_id, subjob_id):
        if not path.name.endswith(".partial") and path.is_file():
            manifest[path.relative_to(root).as_posix()] = path.stat().st_size
    return manifest


def read_done_marker(config, workunit_id, subjob_id, fingerprint):
    """The subjob's .done contents if it is complete for THIS workunit layout,
    otherwise None (missing, unreadable, another wave's, or any output it
    recorded is gone or changed size, e.g. a restore that stopped part way)."""
    marker = read_json(done_marker_path(config, workunit_id, subjob_id))
    if not isinstance(marker, dict) or marker.get('fingerprint') != fingerprint:
        return None
    if not isinstance(marker.get('outputs'), dict) or not outputs_complete(config, workunit_id, subjob_id):
        return None
    root = job_root(config)
    for relative_path, size in marker['outputs'].items():
        try:
            if (root / relative_path).stat().st_size != size:
                return None
        except OSError:
            return None
    return marker


def remove_unrecorded_outputs(config, workunit_id, subjob_id, marker):
    """Delete outputs on disk that the subjob's .done does not record, e.g. a
    failed attempt's table brought back by a restore."""
    recorded = set(marker.get('outputs', {}))
    root = job_root(config)
    for path in subjob_output_paths(config, workunit_id, subjob_id):
        if path.is_file() and path.relative_to(root).as_posix() not in recorded:
            remove_file(path)


def run_started():
    """Whether an afvs_run_local.py has started on this workflow (this wave)."""
    return RUN_STARTED.exists()


def runner_alive():
    """True while an afvs_run_local.py holds the runner lock. The kernel drops
    the lock when that process dies, however it dies."""
    try:
        lock_file = open(RUNNER_LOCK, "r")
    except FileNotFoundError:
        return False
    with lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as err:
            if err.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                return True
            raise
        fcntl.flock(lock_file, fcntl.LOCK_UN)
    return False


def subjob_state(config, workunit_id, subjob_id, subjob, alive=None):
    """(state, marker) for one subjob. `alive` overrides the runner check;
    leave it None so liveness is read AFTER the running marker (the runner
    takes its lock before it writes any).

    SUCCEEDED  valid .done for this layout
    RUNNING    an attempt is in flight under a live runner
    FAILED     the last attempt failed, its runner died mid-attempt, or a run
               ended without reaching it
    PENDING    not attempted yet, and a runner may still get to it
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
        # The attempt may have succeeded (and its runner exited) since the
        # .done was read above
        marker = read_done_marker(config, workunit_id, subjob_id, subjob.get('fingerprint'))
        if marker is not None:
            return SUCCEEDED, marker
        return FAILED, running

    failed = read_json(attempt_path(workunit_id, subjob_id, "failed"))
    if failed is not None:
        return FAILED, failed

    # Never attempted. Once a run has started and no runner is left, nothing
    # will ever pick it up, so it is reported FAILED (and recovery runs it)
    # rather than PENDING for good.
    if run_started():
        if alive is None:
            alive = runner_alive()
        if not alive:
            # It may have been run, and finished, since the reads above
            marker = read_done_marker(config, workunit_id, subjob_id, subjob.get('fingerprint'))
            if marker is not None:
                return SUCCEEDED, marker
            return FAILED, None

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
