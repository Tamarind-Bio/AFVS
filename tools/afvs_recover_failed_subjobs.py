#!/usr/bin/env python3

# Copyright (C) 2025 Hammad Khan
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
# Description: Identify and resubmit failed subjobs to an on-demand queue
#
# ---------------------------------------------------------------------------

import os
import json
import time
import secrets
import argparse
import sys
import tempfile
import tarfile
import shutil
from pathlib import Path
import afvs_presign

MAX_LIGANDS_ON_DEMAND = 100000 # This will be about 100 subjobs

def parse_config(filename):
    with open(filename, "r") as read_file:
        config = json.load(read_file)
    return config


def get_subjob_collections_from_workunit(ctx, workunit_id, subjob_id, workunit_data):
    """
    Download and extract collection data from original workunit tarball
    """
    # Download the workunit tarball from S3
    if 's3_download_path' in workunit_data:
        s3_path = workunit_data['s3_download_path']
        bucket = ctx['config']['object_store_job_bucket']

        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
            try:
                ctx['s3'].download_file(bucket, s3_path, tmp_file.name)

                # Extract config.json from tarball
                with tarfile.open(tmp_file.name, 'r:gz') as tar:
                    config_member = tar.getmember('vf_input/config.json')
                    config_file = tar.extractfile(config_member)
                    workunit_config = json.load(config_file)

                    # Get the subjob's collections
                    if subjob_id in workunit_config['subjobs']:
                        return workunit_config['subjobs'][subjob_id].get('collections', {})

            except Exception as e:
                print(f"  WARNING: Could not retrieve collections for workunit {workunit_id}, subjob {subjob_id}: {e}")
                return {}
            finally:
                os.unlink(tmp_file.name)

    return {}


def identify_failed_subjobs(status_file, ctx):
    """
    Identify all failed subjobs from the status.json file

    Returns:
        List of dicts containing workunit_id, subjob_id, and subjob data with collections
    """
    with open(status_file, "r") as read_file:
        status = json.load(read_file)

    failed_subjobs = []

    for workunit_id, workunit in status['workunits'].items():
        if 'subjobs' not in workunit:
            continue

        for subjob_id, subjob in workunit['subjobs'].items():
            if 'status' in subjob and subjob['status'] == 'FAILED':
                # Get collections from original workunit tarball
                collections = get_subjob_collections_from_workunit(ctx, workunit_id, subjob_id, workunit)

                failed_subjobs.append({
                    'workunit_id': workunit_id,
                    'subjob_id': subjob_id,
                    'subjob': subjob,
                    'collections': collections,  # Add collections here
                    'workunit': workunit
                })

    return failed_subjobs


def create_recovery_workunit(ctx, failed_subjobs, recovery_workunit_id):
    """
    Create a new workunit tarball containing only the failed subjobs

    Returns:
        Dict with recovery workunit information
    """
    temp_path = ctx['config'].get('tempdir_default', '')
    if temp_path and temp_path != "":
        temp_path = os.path.join(temp_path, '')
    else:
        temp_path = None

    # Create temporary directories
    temp_dir = tempfile.TemporaryDirectory(prefix=temp_path)
    temp_dir_tar = tempfile.TemporaryDirectory(prefix=temp_path)

    # Build the recovery subjobs structure
    recovery_subjobs = {}
    for idx, failed_item in enumerate(failed_subjobs):
        # Skip if no collections data available
        if not failed_item['collections']:
            print(f"  WARNING: Skipping {failed_item['workunit_id']}:{failed_item['subjob_id']} - no collections data")
            continue

        # Map old subjob to new index in recovery workunit
        recovery_subjobs[str(idx)] = {
            'collections': failed_item['collections'],
            'ligands_expected': failed_item['subjob'].get('ligands_expected', 0),
            'original_workunit_id': failed_item['workunit_id'],
            'original_subjob_id': failed_item['subjob_id']
        }

    # Write out the config JSON information
    output_structure = {
        'config': ctx['config'],
        'subjobs': recovery_subjobs,
        'recovery_mode': True
    }

    with open(f'{temp_dir.name}/config.json', 'w') as json_out:
        json.dump(output_structure, json_out, indent=4)

    # Copy input files directory
    shutil.copytree(
        f"{ctx['config']['docking_scenario_basefolder']}",
        f"{temp_dir.name}/input-files"
    )

    # Generate the tarball
    out = tarfile.open(f'{temp_dir_tar.name}/{recovery_workunit_id}.tar.gz', mode='x:gz')
    out.add(temp_dir.name, arcname="vf_input")
    out.close()

    # Upload to S3 or copy to shared filesystem (presigned keeps the job in S3)
    if ctx['config']['job_storage_mode'] in ("s3", "presigned"):
        object_path = [
            ctx['config']['object_store_job_prefix'],
            "recovery",
            ctx['config']['job_letter'],
            "input",
            "tasks",
            f"{recovery_workunit_id}.tar.gz"
        ]
        object_name = "/".join(object_path)

        try:
            response = ctx['s3'].upload_file(
                f'{temp_dir_tar.name}/{recovery_workunit_id}.tar.gz',
                ctx['config']['object_store_job_bucket'],
                object_name
            )
        except Exception as e:
            print(f"Error uploading recovery workunit: {e}")
            raise

        temp_dir.cleanup()
        temp_dir_tar.cleanup()

        return {
            'subjobs': {k: {'ligands_expected': v['ligands_expected']} for k, v in recovery_subjobs.items()},
            's3_download_path': object_name,
            'recovery_subjobs_mapping': {k: {'original_workunit_id': v['original_workunit_id'],
                                             'original_subjob_id': v['original_subjob_id']}
                                        for k, v in recovery_subjobs.items()}
        }

    elif ctx['config']['job_storage_mode'] == "sharedfs":
        sharedfs_workunit_path = Path(ctx['config']['sharedfs_workunit_path']) / "recovery" / f"{recovery_workunit_id}.tar.gz"
        sharedfs_workunit_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f'{temp_dir_tar.name}/{recovery_workunit_id}.tar.gz', sharedfs_workunit_path)

        temp_dir.cleanup()
        temp_dir_tar.cleanup()

        return {
            'subjobs': {k: {'ligands_expected': v['ligands_expected']} for k, v in recovery_subjobs.items()},
            'download_path': sharedfs_workunit_path.as_posix(),
            'recovery_subjobs_mapping': {k: {'original_workunit_id': v['original_workunit_id'],
                                             'original_subjob_id': v['original_subjob_id']}
                                        for k, v in recovery_subjobs.items()}
        }


def submit_recovery_job(ctx, recovery_workunit, recovery_id, queue_name=None):
    """
    Submit recovery workunit to AWS Batch (optionally to a specific queue)
    """
    import botocore

    config = ctx['config']
    client = ctx['batch']

    subjobs_count = len(recovery_workunit['subjobs'])

    # AWS Batch doesn't allow array of 1
    if subjobs_count == 1:
        subjobs_count = 2

    # Determine queue to use
    if queue_name:
        job_queue = queue_name
    else:
        # Default to queue1 for recovery
        job_queue = f"{config['aws_batch_prefix']}-queue1"

    # Determine storage path
    if 's3_download_path' in recovery_workunit:
        storage_path = recovery_workunit['s3_download_path']
        storage_mode = "s3"
        bucket = config['object_store_job_bucket']
    else:
        storage_path = recovery_workunit['download_path']
        storage_mode = "sharedfs"
        bucket = ""

    try:
        response = client.submit_job(
            jobName=f'afvs-{config["job_letter"]}-recovery-{recovery_id}',
            timeout={
                'attemptDurationSeconds': int(config["aws_batch_subjob_timeout"])
            },
            jobQueue=job_queue,
            arrayProperties={
                'size': subjobs_count
            },
            jobDefinition=f"{config['aws_batch_jobdef']}",
            containerOverrides={
                'resourceRequirements': [
                    {
                        'type': 'VCPU',
                        'value': config['aws_batch_subjob_vcpus'],
                    },
                    {
                        'type': 'MEMORY',
                        'value': config['aws_batch_subjob_memory'],
                    },
                ],
                'environment': [
                    {
                        'name': 'AFVS_RUN_MODE',
                        'value': "awsbatch"
                    },
                    {
                        'name': 'AFVS_JOB_STORAGE_MODE',
                        'value': storage_mode
                    },
                    {
                        'name': 'AFVS_VCPUS',
                        'value': config['threads_to_use']
                    },
                    {
                        'name': 'AFVS_TMP_PATH',
                        'value': config['tempdir_default']
                    },
                    {
                        'name': 'AFVS_RUN_SEQUENTIAL',
                        'value': "0"
                    },
                    {
                        'name': 'AFVS_WORKUNIT',
                        'value': f"recovery-{recovery_id}"
                    },
                    {
                        'name': 'AFVS_CONFIG_JOB_OBJECT',
                        'value': storage_path
                    },
                    {
                        'name': 'AFVS_CONFIG_JOB_BUCKET',
                        'value': bucket
                    },
                ]
            }
        )

        return {
            'status': 'SUBMITTED',
            'job_id': response['jobId'],
            'queue': job_queue,
            'subjobs_count': len(recovery_workunit['subjobs'])
        }

    except botocore.exceptions.ClientError as error:
        print(f"Error submitting recovery job: {error}")
        raise


def latest_subjob_states(status, recovery_status):
    """(workunit_id, subjob_id) of every original subjob -> where its latest
    attempt: (state, tarball key, subjob id in that tarball, subjob entry,
    round), from the latest recovery round that included it, else the main
    run (round 0)."""
    now = time.time()
    latest = {}
    for workunit_id, workunit in status['workunits'].items():
        for subjob_id, subjob in workunit.get('subjobs', {}).items():
            latest[(workunit_id, subjob_id)] = (subjob.get('status'), workunit.get('s3_download_path'), subjob_id, subjob, 0)

    for recovery_id, recovery in sorted(recovery_status['recovery_workunits'].items(),
                                        key=lambda item: item[1].get('round', 0)):
        # Abandoned, or recorded and not seen by Batch for too long (only
        # under --report-only, which resolves nothing): its subjobs' previous
        # attempt stands, as in afvs_get_status.py
        if recovery.get('abandoned') or (not recovery.get('job_id')
                                         and not afvs_presign.pending_in_grace(recovery, now)):
            continue
        for index, mapping in recovery.get('recovery_subjobs_mapping', {}).items():
            subjob = recovery['subjobs'].get(index, {})
            # Not yet seen by afvs_get_status.py: submitted, not failed
            latest[(mapping['original_workunit_id'], mapping['original_subjob_id'])] = (
                subjob.get('status', "SUBMITTED"), recovery.get('s3_download_path'), index, subjob,
                recovery.get('round', 0))
    return latest


def publish_recovery_manifest(ctx, recovery_workunit, recovery_id, tarball_subjobs):
    """Sign and upload a recovery workunit's manifest; its presigned GET."""
    return afvs_presign.publish_manifest(
        ctx['s3'], ctx['config'], f"recovery-{recovery_id}", recovery_workunit['s3_download_path'], tarball_subjobs)


def submit_presigned_recovery_job(ctx, recovery_workunit, recovery_id, manifest_url, job_name):
    """Submit a recovery workunit whose job, like any presigned job, has no
    AWS identity: it gets the presigned GET of a freshly signed manifest."""
    config = ctx['config']
    workunit_id = f"recovery-{recovery_id}"

    subjobs_count = len(recovery_workunit['subjobs'])

    # The sealed queue (aws_batch_prefix=vf-sealed -> vf-sealed-queue1); --queue
    # names the legacy on-demand queue and does not apply here
    job_queue = f"{config['aws_batch_prefix']}-queue1"

    response = ctx['batch'].submit_job(
        jobName=job_name,
        timeout={
            'attemptDurationSeconds': int(config["aws_batch_subjob_timeout"])
        },
        jobQueue=job_queue,
        arrayProperties={
            'size': 2 if subjobs_count == 1 else subjobs_count
        },
        jobDefinition=f"{config['aws_batch_jobdef']}",
        containerOverrides={
            'resourceRequirements': [
                {'type': 'VCPU', 'value': config['aws_batch_subjob_vcpus']},
                {'type': 'MEMORY', 'value': config['aws_batch_subjob_memory']},
            ],
            'environment': afvs_presign.job_environment(config, workunit_id, manifest_url)
        }
    )

    return {
        'status': 'SUBMITTED',
        'job_id': response['jobId'],
        'queue': job_queue,
        'subjobs_count': subjobs_count
    }


def recover_presigned(args, config):
    """job_storage_mode=presigned: resubmit every subjob whose LATEST attempt
    failed (a later recovery round overrides the main run and earlier rounds),
    as a new round appended to status.recovery.json, with freshly signed URLs."""
    with afvs_presign.recovery_lock():
        return _recover_presigned(args, config)


def save_recovery_status(recovery_status_file, recovery_status):
    with open(f"{recovery_status_file}.tmp", 'w') as f:
        json.dump(recovery_status, f, indent=4)
    os.replace(f"{recovery_status_file}.tmp", recovery_status_file)


def find_batch_jobs(ctx, queue, job_name):
    """Every job named job_name on queue, all pages."""
    jobs = []
    kwargs = {'jobQueue': queue, 'filters': [{'name': 'JOB_NAME', 'values': [job_name]}]}
    while True:
        response = ctx['batch'].list_jobs(**kwargs)
        jobs.extend(response.get('jobSummaryList', []))
        if not response.get('nextToken'):
            break
        kwargs['nextToken'] = response['nextToken']
    return sorted(jobs, key=lambda job: job.get('createdAt', 0))


def resolve_pending_rounds(ctx, recovery_status_file, recovery_status):
    """A recovery workunit is recorded (job_id None) before it is submitted.
    If an earlier call died between the two, find out from Batch whether the
    job exists (its name is unique to that attempt): keep it if so. If not,
    wait out ListJobs' lag, then mark it abandoned. It stays in the file so its id (and so its tarball and manifest
    keys) is never reused, and its subjobs count as failed again."""
    changed = False
    for recovery_id, recovery in recovery_status['recovery_workunits'].items():
        if recovery.get('job_id') or recovery.get('abandoned'):
            continue
        jobs = find_batch_jobs(ctx, recovery['queue'], recovery['job_name'])
        if jobs:
            recovery['job_id'] = jobs[0]['jobId']
            print(f"  Found recovery workunit {recovery_id} already submitted as job {recovery['job_id']}")
            if len(jobs) > 1:
                print(f"  WARNING: {len(jobs)} jobs named {recovery['job_name']}; only the first is tracked")
        elif afvs_presign.pending_in_grace(recovery, time.time()):
            print(f"  Recovery workunit {recovery_id} is not listed by Batch yet; its subjobs count as submitted")
            continue
        else:
            recovery['abandoned'] = True
            print(f"  Recovery workunit {recovery_id} was never submitted; its subjobs are retried")
        changed = True
    if changed:
        save_recovery_status(recovery_status_file, recovery_status)


def _recover_presigned(args, config):
    if args.max_subjobs is not None and args.max_subjobs < 1:
        print("--max-subjobs must be at least 1")
        return 2
    if args.from_round is not None and args.from_round < 1:
        print("--from-round must be at least 1")
        return 2

    import boto3
    import botocore.exceptions
    from botocore.config import Config

    ctx = {
        'config': config,
        's3': afvs_presign.s3_client(config),
        'batch': boto3.client('batch', config=Config(region_name=config['aws_region'])),
        'client_error': botocore.exceptions.ClientError,
    }

    with open(args.status_file, "r") as read_file:
        status = json.load(read_file)

    recovery_status_file = "../workflow/status.recovery.json"
    recovery_status = {'recovery_workunits': {}}
    if os.path.exists(recovery_status_file):
        with open(recovery_status_file, "r") as read_file:
            recovery_status = json.load(read_file)
    # --report-only changes nothing, so a pending record is left as it is
    if not args.report_only:
        resolve_pending_rounds(ctx, recovery_status_file, recovery_status)

    print("Analyzing status files for subjobs whose latest attempt failed...")
    tarballs = {}
    failed_subjobs = []
    for (workunit_id, subjob_id), (state, tarball_key, tarball_subjob_id, subjob, attempt_round) in latest_subjob_states(status, recovery_status).items():
        if state != "FAILED":
            continue
        # --from-round R: only failures from before round R. A subjob that
        # failed again in round R waits for round R+1, so it cannot crowd out
        # this round's own starting failures.
        if args.from_round is not None and attempt_round >= args.from_round:
            continue
        if tarball_key not in tarballs:
            tarballs[tarball_key] = afvs_presign.tarball_subjobs(ctx['s3'], config, tarball_key)
        failed_subjobs.append({
            'workunit_id': workunit_id,
            'subjob_id': subjob_id,
            'subjob': status['workunits'][workunit_id]['subjobs'][subjob_id],
            'collections': tarballs[tarball_key].get(tarball_subjob_id, {}).get('collections', {}),
            'workunit': status['workunits'][workunit_id],
            'attempt_round': attempt_round,
        })

    if not failed_subjobs:
        print("No failed subjobs found!")
        print("Remaining failed subjobs: 0")
        return 0

    # Recovery workunits are numbered densely from their collections: a
    # subjob without any (only a malformed tarball has none) cannot be rerun
    missing = [item for item in failed_subjobs if not item['collections']]
    if missing:
        for item in missing:
            print(f"ERROR: no collections recorded for {item['workunit_id']}:{item['subjob_id']}")
        return 1

    # A window of at most --max-subjobs per call, oldest failure first, then
    # in a stable order. What is submitted now is in this round, so its latest
    # state is no longer FAILED and the next call carries on with the rest; a
    # subjob that fails again goes behind those that have waited longer.
    failed_subjobs.sort(key=lambda item: (item['attempt_round'], int(item['workunit_id']), int(item['subjob_id'])))
    remaining = []
    if args.max_subjobs is not None:
        failed_subjobs, remaining = failed_subjobs[:args.max_subjobs], failed_subjobs[args.max_subjobs:]

    print(f"\nFound {len(failed_subjobs) + len(remaining)} failed subjobs, submitting {len(failed_subjobs)}:")
    for item in failed_subjobs:
        print(f"{item['workunit_id']:<15} {item['subjob_id']:<12} {item['subjob'].get('ligands_expected', 0):<20}")

    if args.report_only:
        print("\n[Report-only mode: No resubmission performed]")
        print(f"Remaining failed subjobs: {len(failed_subjobs) + len(remaining)}")
        return 0

    # Every call with the same --from-round R belongs to round R; without it,
    # each call is a new round
    if args.from_round is not None:
        recovery_round = args.from_round
    else:
        recovery_round = 1 + max([entry.get('round', 0) for entry in recovery_status['recovery_workunits'].values()] or [0])
    already_in_round = sum(1 for entry in recovery_status['recovery_workunits'].values() if entry.get('round') == recovery_round)
    print(f"\nCreating recovery round {recovery_round} (max {args.max_subjobs_per_workunit} subjobs per workunit)...")

    for i in range(0, len(failed_subjobs), args.max_subjobs_per_workunit):
        batch = failed_subjobs[i:i + args.max_subjobs_per_workunit]
        already_in_round += 1
        recovery_id = f"{config['job_letter']}-r{recovery_round}-{already_in_round}"

        recovery_workunit = create_recovery_workunit(ctx, batch, recovery_id)
        # The same indexing create_recovery_workunit uses (items without
        # collections are skipped, keeping their index)
        tarball_subjobs = {str(index): {'collections': item['collections']}
                           for index, item in enumerate(batch) if item['collections']}

        # Recorded BEFORE it is submitted (job_id None), so a crash between
        # the two leaves a trace resolve_pending_rounds can settle; saved
        # again with the job id right after
        # The random suffix makes the job name unique to this attempt, so
        # resolve_pending_rounds cannot adopt another job of the same name
        entry = {
            'round': recovery_round,
            'job_id': None,
            'recorded_at': time.time(),
            'job_name': f'afvs-r-{recovery_id}-{secrets.token_hex(4)}',
            'queue': f"{config['aws_batch_prefix']}-queue1",
            'subjobs_count': len(recovery_workunit['subjobs']),
            's3_download_path': recovery_workunit['s3_download_path'],
            'subjobs': recovery_workunit['subjobs'],
            'recovery_subjobs_mapping': recovery_workunit['recovery_subjobs_mapping']
        }
        recovery_status['recovery_workunits'][recovery_id] = entry
        save_recovery_status(recovery_status_file, recovery_status)

        try:
            manifest_url = publish_recovery_manifest(ctx, recovery_workunit, recovery_id, tarball_subjobs)
        except BaseException:
            # Batch was never called: there is no job, so the next call
            # retries its subjobs at once
            entry['abandoned'] = True
            save_recovery_status(recovery_status_file, recovery_status)
            raise
        try:
            submission_result = submit_presigned_recovery_job(ctx, recovery_workunit, recovery_id, manifest_url, entry['job_name'])
        except ctx['client_error'] as error:
            # Batch rejected the request itself (ClientException, HTTP 400):
            # there is no job, so the next call retries its subjobs at once.
            # Anything else (throttling, a 5xx after a retry whose first
            # attempt may have gone through) stays pending for
            # resolve_pending_rounds to adopt or abandon
            if getattr(error, 'response', {}).get('Error', {}).get('Code') == 'ClientException':
                entry['abandoned'] = True
                save_recovery_status(recovery_status_file, recovery_status)
            raise error
        entry['job_id'] = submission_result['job_id']
        save_recovery_status(recovery_status_file, recovery_status)
        print(f"  Submitted recovery workunit {recovery_id} as job {submission_result['job_id']}")

    print(f"\nRecovery status saved to {recovery_status_file}")
    print(f"Remaining failed subjobs: {len(remaining)}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Identify and resubmit failed subjobs from AFVS workflow'
    )
    parser.add_argument(
        '--status-file',
        default='../workflow/status.json',
        help='Path to status.json file (default: ../workflow/status.json)'
    )
    parser.add_argument(
        '--config-file',
        default='../workflow/config.json',
        help='Path to config.json file (default: ../workflow/config.json)'
    )
    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Only report failed subjobs without resubmitting'
    )
    parser.add_argument(
        '--queue',
        help='AWS Batch queue name to submit recovery jobs (e.g., vf-queue-ondemand). If not specified, uses default spot queue.'
    )
    parser.add_argument(
        '--max-subjobs',
        type=int,
        help='job_storage_mode=presigned: submit at most this many failed subjobs per call (repeat calls continue)'
    )
    parser.add_argument(
        '--from-round',
        type=int,
        help='job_storage_mode=presigned: this call is part of recovery round R and takes only subjobs whose latest failure is from before R'
    )
    parser.add_argument(
        '--max-subjobs-per-workunit',
        type=int,
        default=100,
        help='Maximum number of failed subjobs to group into a single recovery workunit (default: 100)'
    )

    args = parser.parse_args()

    # Parse configuration
    config = parse_config(args.config_file)

    if afvs_presign.is_presigned(config):
        return recover_presigned(args, config)

    import boto3
    from botocore.config import Config

    # Initialize AWS clients (needed to download workunit tarballs)
    aws_config = Config(region_name=config['aws_region'])
    ctx = {
        'config': config,
        's3': boto3.client('s3', config=aws_config),
        'batch': boto3.client('batch', config=aws_config)
    }

    # Identify failed subjobs
    print("Analyzing status file for failed subjobs...")
    print("Downloading workunit tarballs to extract collection data...")
    failed_subjobs = identify_failed_subjobs(args.status_file, ctx)

    if not failed_subjobs:
        print("No failed subjobs found!")
        return 0

    print(f"\nFound {len(failed_subjobs)} failed subjobs:")
    print(f"{'Workunit ID':<15} {'Subjob ID':<12} {'Expected Ligands':<20} {'Collections'}")
    print("-" * 80)

    total_expected_ligands = 0
    for item in failed_subjobs:
        if item['collections']:
            collections_str = ', '.join(item['collections'].keys())
            if len(collections_str) > 35:
                collections_str = collections_str[:32] + "..."
        else:
            collections_str = "N/A (no collections data)"

        expected = item['subjob'].get('ligands_expected', 0)
        total_expected_ligands += expected

        print(f"{item['workunit_id']:<15} {item['subjob_id']:<12} {expected:<20} {collections_str}")

    print(f"\nTotal expected ligands in failed subjobs: {total_expected_ligands:,}")

    if args.report_only:
        print("\n[Report-only mode: No resubmission performed]")
        return 0

    if total_expected_ligands > MAX_LIGANDS_ON_DEMAND and args.queue == 'vf-queue-ondemand':
        print(f"\nWARNING: Total expected ligands ({total_expected_ligands:,}) exceeds on-demand threshold ({MAX_LIGANDS_ON_DEMAND:,})")
        print(f"Automatically switching to spot queues to reduce costs.")
        print(f"To override, increase MAX_LIGANDS_ON_DEMAND in the script.\n")
        args.queue = None

    # Create recovery workunits
    print(f"\nCreating recovery workunits (max {args.max_subjobs_per_workunit} subjobs per workunit)...")

    recovery_workunits = []
    recovery_status = {'recovery_workunits': {}}

    for i in range(0, len(failed_subjobs), args.max_subjobs_per_workunit):
        batch = failed_subjobs[i:i + args.max_subjobs_per_workunit]
        recovery_id = f"{config['job_letter']}-{i // args.max_subjobs_per_workunit + 1}"

        print(f"  Creating recovery workunit {recovery_id} with {len(batch)} subjobs...")
        recovery_workunit = create_recovery_workunit(ctx, batch, recovery_id)

        print(f"  Submitting recovery workunit {recovery_id} to AWS Batch...")
        submission_result = submit_recovery_job(ctx, recovery_workunit, recovery_id, args.queue)

        recovery_workunits.append({
            'recovery_id': recovery_id,
            'workunit': recovery_workunit,
            'submission': submission_result
        })

        recovery_status['recovery_workunits'][recovery_id] = {
            'job_id': submission_result['job_id'],
            'queue': submission_result['queue'],
            'subjobs_count': submission_result['subjobs_count'],
            'subjobs': recovery_workunit['subjobs'],
            'recovery_subjobs_mapping': recovery_workunit['recovery_subjobs_mapping']
        }

        print(f"  Submitted as job {submission_result['job_id']} to queue {submission_result['queue']}")

    # Save recovery status
    recovery_status_file = "../workflow/status.recovery.json"
    with open(recovery_status_file, 'w') as f:
        json.dump(recovery_status, f, indent=4)

    print(f"\nRecovery status saved to {recovery_status_file}")
    print(f"\nSuccessfully created and submitted {len(recovery_workunits)} recovery workunit(s)")
    print(f"  Total failed subjobs being recovered: {len(failed_subjobs)}")
    print(f"  Queue used: {recovery_workunits[0]['submission']['queue']}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
