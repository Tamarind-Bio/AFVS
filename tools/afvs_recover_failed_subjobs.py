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
import boto3
import botocore
import argparse
import sys
import tempfile
import tarfile
import shutil
from pathlib import Path
from botocore.config import Config

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

    # Upload to S3 or copy to shared filesystem
    if ctx['config']['job_storage_mode'] == "s3":
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
        '--max-subjobs-per-workunit',
        type=int,
        default=100,
        help='Maximum number of failed subjobs to group into a single recovery workunit (default: 100)'
    )

    args = parser.parse_args()

    # Parse configuration
    config = parse_config(args.config_file)

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
