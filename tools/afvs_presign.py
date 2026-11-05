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
# Description: Head-side signing for job_storage_mode=presigned
#
# Docking jobs in presigned mode have no AWS identity. For each workunit the
# head (afvs_submit_jobs.py, afvs_recover_failed_subjobs.py) writes a manifest
# with presigned GET URLs for the workunit tarball and its collections and
# presigned PUT URLs for every output at the usual keys, uploads it, and gives
# the job one presigned GET for it (AFVS_CONFIG_JOB_URL).
#
# Signing uses the standard AWS credentials in this process's environment.
# A presigned URL is a credential: nothing here prints or logs one.
#
# ---------------------------------------------------------------------------

import contextlib
import fcntl
import io
import json
import tarfile


RECOVERY_LOCK = "../workflow/status.recovery.lock"


@contextlib.contextmanager
def recovery_lock():
    """Held by afvs_recover_failed_subjobs.py for a whole call and by
    afvs_get_status.py while it writes status.recovery.json back, so no
    round is lost and no window is submitted twice."""
    with open(RECOVERY_LOCK, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


MANIFEST_VERSION = 1


def is_presigned(config):
    return config.get('job_storage_mode') == "presigned"


def s3_client(config):
    import boto3
    from botocore.config import Config

    # Submit records job ids only once its whole loop is done, so a transient
    # S3 error must be retried, not end the loop
    return boto3.client('s3', config=Config(region_name=config['aws_region'], signature_version='s3v4',
                                            retries={'mode': 'standard', 'max_attempts': 10}))


def expiry_seconds(config):
    return int(config['presigned_url_expiry_seconds'])


def job_prefix(config):
    # The same prefix afvs_run.generate_output_path uses for s3 outputs
    return f"{config['object_store_job_prefix']}/{config['job_name']}"


def manifest_key(config, workunit_id):
    # Beside the job's prefix, not in it: a manifest grants write access to
    # the job's outputs, so whatever copies or reads <prefix>/<job>/ must not
    # pick manifests up with them
    return f"{config['object_store_job_prefix']}/_manifests/{config['job_name']}/{workunit_id}.json"


def output_keys(config, workunit_id, subjob_id):
    """Every object afvs_run.py writes for one subjob, at the keys
    generate_output_path gives them."""
    prefix = job_prefix(config)
    keys = [f"{prefix}/summary/{workunit_id}/{subjob_id}.json.gz"]
    summary_formats = config.get('summary_formats') or []
    for scenario_key in config['docking_scenario_names']:
        keys.append(f"{prefix}/{scenario_key}/logs/{workunit_id}/{subjob_id}.tar.gz")
        if 'parquet' in summary_formats:
            keys.append(f"{prefix}/{scenario_key}/parquet/{workunit_id}/{subjob_id}.parquet")
        if 'csv.gz' in summary_formats:
            keys.append(f"{prefix}/{scenario_key}/csv/{workunit_id}/{subjob_id}.csv.gz")
    return keys


def sign(s3, config, method, bucket, key):
    client_method = {'GET': 'get_object', 'PUT': 'put_object'}[method]
    return s3.generate_presigned_url(client_method, Params={'Bucket': bucket, 'Key': key},
                                     ExpiresIn=expiry_seconds(config))


def job_environment(config, workunit_id, manifest_url):
    """The whole environment of a presigned docking job. TMPDIR points
    every temporary file at the scratch area: the job's root filesystem is
    read-only."""
    return [
        {'name': 'AFVS_RUN_MODE', 'value': "awsbatch"},
        {'name': 'AFVS_JOB_STORAGE_MODE', 'value': "presigned"},
        {'name': 'AFVS_VCPUS', 'value': config['threads_to_use']},
        {'name': 'AFVS_TMP_PATH', 'value': config['tempdir_default']},
        {'name': 'TMPDIR', 'value': config['tempdir_default']},
        {'name': 'AFVS_RUN_SEQUENTIAL', 'value': "0"},
        {'name': 'AFVS_WORKUNIT', 'value': str(workunit_id)},
        {'name': 'AFVS_CONFIG_JOB_URL', 'value': manifest_url},
    ]


def tarball_subjobs(s3, config, tarball_key):
    """The per-subjob collections recorded in a workunit tarball's config."""
    buffer = io.BytesIO()
    s3.download_fileobj(config['object_store_job_bucket'], tarball_key, buffer)
    buffer.seek(0)
    with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
        return json.load(tar.extractfile("vf_input/config.json"))['subjobs']


def publish_manifest(s3, config, workunit_id, tarball_key, subjobs):
    """Build, upload and sign the manifest for one workunit. `subjobs` maps
    subjob id to its tarball subjob entry (with 'collections'). Returns the
    presigned GET for the manifest, the job's AFVS_CONFIG_JOB_URL."""
    job_bucket = config['object_store_job_bucket']

    manifest = {
        'version': MANIFEST_VERSION,
        'workunit': str(workunit_id),
        'tarball_url': sign(s3, config, 'GET', job_bucket, tarball_key),
        'subjobs': {},
        # A Batch array of one is submitted as two; only then may the job
        # whose subjob is missing treat itself as the padding
        'padded': len(subjobs) == 1,
    }
    for subjob_id, subjob in subjobs.items():
        manifest['subjobs'][str(subjob_id)] = {
            'collections': {
                collection_key: sign(s3, config, 'GET', collection['s3_bucket'], collection['s3_download_path'])
                for collection_key, collection in subjob['collections'].items()
            },
            'outputs': {
                key: sign(s3, config, 'PUT', job_bucket, key)
                for key in output_keys(config, workunit_id, subjob_id)
            },
        }

    key = manifest_key(config, workunit_id)
    s3.put_object(Bucket=job_bucket, Key=key, Body=json.dumps(manifest).encode(),
                  ContentType="application/json")
    return sign(s3, config, 'GET', job_bucket, key)
