# Presigned object-store mode

`job_storage_mode=presigned` runs docking on AWS Batch with jobs that have
**no AWS identity**. The head (the process that runs prepare, submit, status
and recover) signs URLs. Each docking job reaches S3 only through those URLs:
no boto3, no credentials, and it works as UID 1000 on a read-only root
filesystem.

## Configuration (`templates/all.ctrl`)

```
batchsystem=awsbatch
job_storage_mode=presigned
data_storage_mode=s3
presigned_url_expiry_seconds=3300
aws_batch_subjob_timeout=2700
aws_batch_prefix=vf-sealed
aws_batch_number_of_queues=1
aws_batch_jobdef=vf-jobdef-afvs-sealed
```

The `object_store_*`, `aws_region` and other `aws_batch_*` keys are the same as
in s3 mode. Prepare rejects presigned mode unless `batchsystem=awsbatch` and
`data_storage_mode=s3`. It requires `presigned_url_expiry_seconds` to be at
most 604800 (the SigV4 limit) and at least `aws_batch_subjob_timeout` + 300.
URLs are signed at submit, and a job uploads its outputs at the end of its run,
so they must outlast the run plus a little time in the queue. A URL also dies
with the AWS session that signed it, and the production head's sessions last an
hour, so its expiry is at most about 3300 s. A job that starts after its URLs
expired fails its first download, and recovery re-signs and reruns it.
`job_name` must be set (`job_letter` is not supported) and can be at most 100
letters, digits, `-` or `_`, so recovery job names (`afvs-r-<job_name>-r<round>-<n>-<8 hex>`)
stay within Batch's 128 while round and workunit numbers have 9 digits
between them.

## What the head does

- **`afvs_prepare_workunits.py`** uploads each workunit tarball to
  `<prefix>/<job>/input/tasks/<wu>.tar.gz`, as in s3 mode.
- **`afvs_submit_jobs.py`** (and **`afvs_recover_failed_subjobs.py`** for each
  recovery workunit) builds the workunit's manifest:
  - a GET URL for the tarball
  - for each subjob, a GET URL per collection and a PUT URL for every output at
    the usual keys (`summary/<wu>/<sj>.json.gz`, and per scenario
    `logs/…tar.gz`, `parquet/…parquet` and `csv/…csv.gz` as `summary_formats`
    asks)

  It uploads the manifest to `<prefix>/_manifests/<job>/<wu>.json`. That is
  beside the job's output prefix, not inside it, because a manifest grants
  write access to the job's outputs, so whatever copies `<prefix>/<job>/`
  must not pick manifests up. It then submits the job with only
  `AFVS_CONFIG_JOB_URL` (a presigned GET for that manifest) and `TMPDIR` (the
  scratch area, since the job's root filesystem is read-only). It passes no
  tags.
- **Signing** uses the standard AWS credentials in the process environment
  (signature v4). Every URL is valid for `presigned_url_expiry_seconds`, and
  re-running submit or recover re-signs.
- **`afvs_get_status.py`** reads nothing the jobs wrote. Job state and vCPU-hours
  come from Batch `DescribeJobs`. vCPU-hours are the attempts' run time × the
  job's vCPUs, summed over the main run and every recovery round. Docking
  counts are not reported here. A job that DescribeJobs does not return counts
  as gone (its subjobs FAILED, unbilled) only when it was submitted more than
  15 minutes earlier, because DescribeJobs can lag SubmitJob.
- **`afvs_recover_failed_subjobs.py`** reruns every subjob whose latest attempt
  failed, as a new round (`<job>-r<N>-<n>`) appended to `status.recovery.json`.
  A later round overrides earlier rounds and the main run, so a subjob that
  was already recovered is not rerun. `--queue` does not apply: recovery uses
  `<aws_batch_prefix>-queue1`.
- **`afvs_killall_awsbatch.py`** refuses (exit 2). The sealed head has no
  `TerminateJob`, so a stopped screen just stops submitting.
- Any configuration `afvs_get_status.py` does not support exits nonzero.
- `--max-subjobs N` makes recovery submit at most N subjobs per call and print
  `Remaining failed subjobs: K`. Later calls carry on from there, oldest failure
  first, so a subjob that keeps failing cannot hold the others back.
- Submit, recovery and status each hold a lock
  (`../workflow/status.recovery.lock`) for their whole call. Each one reads
  `status.json` and `status.recovery.json` as the others last wrote them.
- If a SubmitJob fails partway through submit, the ids of the jobs already
  submitted are still written, so a rerun skips those jobs.
- Each recovery workunit is recorded before it is submitted, under a job name
  unique to that attempt (`afvs-r-<id>-<random>`).
  - If its manifest cannot be uploaded (Batch is never called), or Batch
    rejects the SubmitJob itself (`ClientException`), the record is marked
    `abandoned` at once. Throttling or a server error leaves the record
    pending, because a retried call's first attempt may have created the job.
  - If a call dies between record and submit, the next call asks Batch
    (`ListJobs`, all pages, by that name) whether the job exists. If it does,
    the call keeps the record. If it doesn't, the record counts as submitted
    for 15 minutes, because ListJobs can lag SubmitJob. After that, the record
    is marked `abandoned` and its subjobs are retried.
  - Status counts a recorded record's subjobs the same way: as submitted for
    those 15 minutes, then as their previous attempt.
  - An abandoned record stays in the file, so its id and S3 keys are never
    reused.

  `--report-only` changes nothing.
- A Batch array of one is submitted as two. The padding child does no work,
  but its run time is billed with its workunit.

## Requirements on the signing role and buckets

- `s3:ListBucket` on the collection prefixes. Without it, S3 answers 403 for a
  missing object and the job cannot tell a missing collection from a rejected
  URL, so the subjob fails instead of recording a failed download.
- The data bucket must be in `aws_region`: every URL is signed for that
  region.
- URLs live as long as the credentials that signed them, so they expire at
  `presigned_url_expiry_seconds` at the latest.

## What a docking job does

The job definition's command is
`["/opt/AFVS/tools/templates/template1.presigned.sh"]`. The image has no
ENTRYPOINT. The script takes the subjob from `AWS_BATCH_JOB_ARRAY_INDEX` and
runs `templates/afvs_run.py`, with `AFVS_JOB_STORAGE_MODE=presigned` and
`AFVS_CONFIG_JOB_URL` set by submit. The job:

1. GETs its manifest, then the workunit tarball.
2. Points each collection and each output at its URL.
3. Removes `AFVS_CONFIG_JOB_URL` from its environment, so no docking program
   inherits it. The process's own `/proc/<pid>/environ` still shows it,
   because that is fixed at exec time.

How it handles failed transfers:

- A 404 on a collection is a failed download, as a missing S3 object is.
- Any other error fails the subjob, so recovery re-signs and reruns it instead
  of silently skipping ligands. That includes an expired or rejected URL, and
  a PUT that is refused.
- Transient errors (5xx, 408, 429, network) are retried with backoff.
- If every downloader has exited, the job fails at once instead of waiting
  for the Batch timeout.

## URLs never reach a log

A presigned URL is a credential. Nothing in this mode prints one:

- Transfer errors report only an HTTP status or an exception type.
- Every other message goes through `redact()`, which removes anything that
  looks like a URL or a signed query string.
- Exceptions are re-raised without their cause.

The tests probe this by forcing an error whose message quotes the URL
(urllib's `InvalidURL`). They assert that no signature token appears in the
job's output or errors.

## Tests

`tests/test_presigned_mode.py` runs a fake S3 on loopback. It checks each URL's
signature and lets each URL do only the method, bucket and key it was signed
for. The head runs against a boto3 stub that mints those URLs, and docking
jobs run as real subprocesses with boto3 unimportable and only loopback
reachable.

It covers submit, docking through the URLs, a failed URL of each kind, the
array-padding child, recovery rounds, killall, and config validation.
`tests/test_presigned_status.py` covers the DescribeJobs-only status and
billing.

## Image

`tools/docker/Dockerfile.sealed` builds the CPU-only amd64 image that both of
these use:

- the Batch docking jobs (job definition `vf-jobdef-afvs-sealed`), which run
  `templates/afvs_run.py`
- the T2 sealed tool container

It has AFVS at `/opt/AFVS` (`AFVS_HOME`), the bundled docking binaries (with
gwovina rebuilt against the image's Boost), Open Babel, pandas/pyarrow and torch/rdkit. It has no ENTRYPOINT, runs as UID 1000,
and has no boto3. It has not been pushed anywhere.

```bash
docker build --platform linux/amd64 -t afvs-sealed -f tools/docker/Dockerfile.sealed .
```

## Running the tests

`afvs_run.py` needs Linux, because `multiprocessing.Queue.qsize()` is not
implemented on macOS. Run the tests in the image:

```bash
printf 'FROM afvs-sealed\nUSER root\nRUN pip install pytest\nUSER 1000\n' | docker build --platform linux/amd64 -t afvs-sealed-test -
docker run --rm --platform linux/amd64 --network none -v "$PWD":/src:ro -w /src/tests \
  afvs-sealed-test python -m pytest -p no:cacheprovider -q
```

`AFVS_TOOLS=<dir>` runs the same suite against another copy of `tools/`. That
is how the golden values in `test_awsbatch_unchanged.py` were checked against
`develop`.

