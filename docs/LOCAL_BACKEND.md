# Local execution backend

Runs a whole screen on one machine (one container), with no AWS Batch and no
S3: the collections are read from a local directory and every output is
written under another. The local code paths do not import boto3 and make no
network calls.

## Configuration (`templates/all.ctrl`)

```
batchsystem=local
job_storage_mode=local
data_storage_mode=local
local_job_root=/out/afvs          # outputs land in <local_job_root>/<job_name>/
local_data_root=/in/collections   # collections, at their S3 keys
local_workers=4                   # subjobs at once (optional, default 1)
local_array_job_size=100          # subjobs per workunit (optional, default 100)
threads_to_use=8                  # docking processes per subjob, as before
```

The three `*=local` settings go together. `object_store_*` and `aws_*` keys
are not needed.

`local_data_root` holds the collection tarballs at the same relative keys they
have in S3 under `object_store_data_collection_prefix`. For example, with
metatranche addressing, `AACAAA_00001` is read from
`<local_data_root>/AA/AACAAA/00001.tar.gz`.

## Running

From `tools/`, as before:

```bash
./afvs_prepare_folders.py --overwrite
./afvs_prepare_workunits.py          # prints "Generated N workunits"
./afvs_run_local.py [--workers N]    # blocks until every subjob has run
./afvs_get_status.py                 # same report as on AWS Batch
./afvs_recover_failed_subjobs.py     # reruns failed subjobs here; --queue is ignored
```

`afvs_run_local.py` exits 0 when every subjob it ran succeeded and 1 when any
failed. If it gets SIGTERM or SIGINT, it stops its subjobs and exits 128+signal.
Only one runner can own a workflow at a time. `afvs_submit_jobs.py` has nothing
to submit in local mode and exits 2.

## Outputs

`<local_job_root>/<job_name>/` has the same layout the S3 backend writes under
`<object_store_job_prefix>/<job_name>/`:

```
<scenario>/parquet/<wu>/<sj>.parquet
<scenario>/csv/<wu>/<sj>.csv.gz
<scenario>/logs/<wu>/<sj>.tar.gz
summary/<wu>/<sj>.json.gz
input/tasks/<wu>.tar.gz
.status/<wu>/<sj>.done
```

Each output file is written to a temporary name and then renamed into place, so
a reader never sees a partial file.

## Completion markers and resume

`.status/<wu>/<sj>.done` is written only after the subjob's `afvs_run.py`
exited 0 and its outputs are in place. It is JSON and records:

- a fingerprint of what the subjob docks
- `vcpu_seconds` (wall time × `threads_to_use`)
- `vcpu_seconds_interrupted`
- `attempts`

A rerun skips every subjob that has a valid marker. A marker is ignored, and
the subjob runs again, when:

- the fingerprint differs. The fingerprint covers:
  - the subjob's collections and counts, so a later active-learning wave that
    reuses the same workunit and subjob ids for different ligands does not
    match
  - the docking scenarios
  - the contents of `docking_scenario_basefolder` (receptor, box), so a run
    under the same `job_name` with another box or receptor does not match
- any output the subjob should have is missing:
  - the summary
  - the log archive of each scenario, when anything was docked
  - the parquet/csv score tables, when a docking succeeded

  This covers a restore that stopped part way.

A missing collection tarball is recorded as a failed download, and the subjob
still completes, as on S3.

To resume on a fresh machine:

1. Restore `<local_job_root>/<job_name>/`, including `.status/`.
2. Run prepare again from the same `todo.all`.
3. Run `./afvs_run_local.py`.

Per-attempt state goes under `../workflow/local/`, which prepare resets:

- running and failed markers
- each subjob's `afvs_run.py` output (`<wu>/<sj>.out`)

## Runner ownership

`afvs_run_local.py` holds an flock on `../workflow/local/runner.lock` while it
runs, and "a runner is alive" means that lock is held. The kernel releases the
lock however the runner dies, so a reused pid can never fake a live runner.

Each subjob runs in its own session:

- When a subjob ends, its whole process group is killed, so worker processes
  cannot outlive an `afvs_run.py` that was OOM-killed.
- A runner that starts after another was killed outright first kills any
  attempt that runner left running, and only then reruns that subjob.

## Status

`afvs_get_status.py` rebuilds its report from the markers on every call. A
subjob is shown as:

| State | When |
| --- | --- |
| `SUCCEEDED` | It has a valid `.done` marker |
| `RUNNING` | An attempt is in flight under a live runner |
| `FAILED` | Its last attempt failed, or its runner died mid-attempt |
| `PENDING` | It has not been attempted yet |

Workunits roll these up the way an AWS Batch array parent does. vCPU hours
come from the `.done` markers, so they include subjobs that finished before a
resume.

## Image

`tools/docker/Dockerfile.local` builds a CPU-only amd64 image with:

- AFVS and the bundled docking binaries
- Open Babel
- python with pandas, pyarrow, torch and rdkit (for active learning)

It does not include boto3. It runs as UID 1000.

The image is laid out for the T2 sealed pod:

- AFVS is installed at `/opt/AFVS`, exported as `AFVS_HOME`.
- There is no ENTRYPOINT. The pod runs `bash run.sh` in `/app`, and `run.sh` calls `$AFVS_HOME/tools/*`.
- Inputs are in `/app/inputs` and outputs go to `/app/out`.

```bash
docker build --platform linux/amd64 -t afvs-local -f tools/docker/Dockerfile.local .
```

## Tests

`tests/` covers the following on a two-collection fixture (a fake docking
engine, plus one run with the real `qvina02`):

- local prepare, run and status
- resume after SIGTERM and after SIGKILL
- recovery
- that the status text parses with the wrapper's awk patterns
- that the AWS Batch / S3 paths are unchanged

`afvs_run.py` needs Linux, because `multiprocessing.Queue.qsize()` is not
implemented on macOS. Run the tests in the image:

```bash
docker build --platform linux/amd64 -t afvs-local -f tools/docker/Dockerfile.local .
printf 'FROM afvs-local\nUSER root\nRUN pip install pytest\nUSER 1000\n' | docker build --platform linux/amd64 -t afvs-local-test -
docker run --rm --platform linux/amd64 --network none -v "$PWD":/src:ro -w /src/tests \
  afvs-local-test python -m pytest -p no:cacheprovider -q
```

The run uses `--network none`. The suite also blocks boto3 imports and inet
connects in every process it starts.
