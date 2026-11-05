#!/usr/bin/env python3


# Copyright (C) 2019 Christoph Gorgulla
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
# Description: Get status of the AWS Batch jobs
#
# Revision history:
# 2021-06-29  Original version
#
# ---------------------------------------------------------------------------


import os
import json
import re
import tempfile
import gzip
import time
import pprint
import hashlib
from pathlib import Path
import argparse
import gzip
import multiprocessing
from multiprocessing import Process
from multiprocessing import Queue
from queue import Empty
import logging
import shutil
import sys
import afvs_presign

# How long a submitted job may be missing from DescribeJobs before it is
# taken as gone (DescribeJobs can lag a SubmitJob)
PARENT_GRACE_SECONDS = 900



def downloader(download_queue, summary_queue, tmp_dir, config):
    import boto3
    import botocore
    from botocore.config import Config

    temp_dir = tempfile.mkdtemp(prefix=tmp_dir)

    botoconfig = Config(
       retries = {
          'max_attempts': 25,
          'mode': 'standard'
       }
    )

    s3 = boto3.client('s3', config=botoconfig)

    while True:
        try:
            item = download_queue.get(timeout=20.5)
        except Empty:
            continue

        if item is None:
            summary_queue.put(None)
            break

        local_path= f"{temp_dir}/tmp.{item['workunit_id']}.{item['subjob_id']}.json.gz"
        remote_path = item['s3_path']
        job_bucket = config['object_store_job_bucket']

        # Fail closed: a summary that cannot be downloaded or parsed goes back
        # with result None (the consumer marks that subjob FAILED) instead of
        # being dropped, which used to count the subjob as succeeded
        item['result'] = None
        try:
            with open(local_path, 'wb') as f:
                s3.download_fileobj(job_bucket, remote_path, f)
            logging.debug(f"downloaded to {local_path}")
        except botocore.exceptions.ClientError as error:
            logging.error(f"Failed to download from S3 {job_bucket}/{remote_path} to {local_path}, ({error})")
        except Exception as error:
            logging.error(f"Failed to download from S3 {job_bucket}/{remote_path} to {local_path}, ({error!r})")
        else:
            try:
                with gzip.open(local_path, "rb") as read_file:
                    item['result'] = check_summary(json.load(read_file))
            except Exception as error:
                logging.error(f"Could not parse the summary {job_bucket}/{remote_path} ({error!r})")

        # Move it to the next step if there's space
        while summary_queue.qsize() > 1000:
            time.sleep(0.2)

        summary_queue.put(item)
        if os.path.exists(local_path):
            os.remove(local_path)


def check_summary(result):
    """The summary fields process() reads, with the types it adds up; raises
    on anything else so a malformed summary is treated as a failed subjob."""

    def count(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"not a count: {value!r}")
        return value

    count(result['total_dockings'])
    count(result['dockings_status']['success'])
    count(result['dockings_status']['failed'])
    count(result['skipped_ligands'])
    count(result['failed_downloads'])
    count(result['failed_downloads_dockings'])
    if not isinstance(result['failed_downloads_log'], list):
        raise ValueError("failed_downloads_log is not a list")
    for log_item in result['failed_downloads_log']:
        for key in ('base_collection_key', 'reason'):
            if not isinstance(log_item[key], str):
                raise ValueError(f"failed_downloads_log {key} is not a string")
        count(log_item['dockings'])
    return result


def parse_config(filename):
    with open(filename, "r") as read_file:
        config = json.load(read_file)

    return config


def process(config):
    import boto3
    from botocore.config import Config

    statuses = ['SUBMITTED','PENDING','RUNNABLE','STARTING','RUNNING','SUCCEEDED','FAILED']


    aws_config = Config(
        region_name=config['aws_region']
    )


    parser = argparse.ArgumentParser()
    parser.add_argument('--detailed', action='store_true',
        help="Get detailed ligand information (this can take a long time!)")
    parser.add_argument('--recovery', action='store_true',
        help="Show only recovery jobs status")
    args = parser.parse_args()

    client = boto3.client('batch', config=aws_config)

    # load the status file that is keeping track of the data
    with open("../workflow/status.json", "r") as read_file:
        complete = json.load(read_file)

    # Determine which status to process
    if args.recovery:
        recovery_status_file = "../workflow/status.recovery.json"
        if os.path.exists(recovery_status_file):
            with open(recovery_status_file, "r") as read_file:
                recovery_complete = json.load(read_file)

            # Use recovery workunits as the main status
            status = recovery_complete['recovery_workunits']
            status_file_path = recovery_status_file
            is_recovery_mode = True
            workunit_prefix = "recovery-"
        else:
            print(f"Error: Recovery status file not found at {recovery_status_file}")
            print(f"No recovery jobs have been submitted yet.")
            return
    else:
        status = complete['workunits']
        status_file_path = "../workflow/status.json"
        is_recovery_mode = False
        workunit_prefix = ""

    finished = []

    use_list_jobs_status = ['SUBMITTED', 'PENDING']
    final_states = ['SUCCEEDED', 'FAILED']

    vcpus_per_job = int(config['aws_batch_subjob_vcpus'])

    # Which ones do we need to check
    workunits_to_check = []
    for workunit_id, workunit in status.items():
        if 'status' not in workunit:
            # Initialize status for workunits that don't have it yet (e.g., newly submitted recovery jobs)
            if 'job_id' in workunit:
                workunit['status'] = 'SUBMITTED'
            else:
                # workunit has not been submitted yet
                continue

        if(workunit['status'] in use_list_jobs_status):
            workunits_to_check.append({
                    'workunit_id': workunit_id,
                    'batch_jobid': workunit['job_id']
                }
                )

    workunits_to_recompute = {}
    subjobs_to_check = []
    failed_downloads = []

    print("Getting workunit status")

    # Check the array parents
    for status_index_begin in range(0, len(workunits_to_check), 100):

        job_keys_to_check = []
        job_key_mapping = {}

        for status_item in workunits_to_check[status_index_begin:(status_index_begin+100)]:

            workunit_id = status_item['workunit_id']
            batch_jobid = status_item['batch_jobid']

            job_keys_to_check.append(batch_jobid)
            job_key_mapping[batch_jobid] = { 'workunit_id': workunit_id }

        response = client.describe_jobs(jobs=job_keys_to_check)

        if 'jobs' in response:

            for job in response['jobs']:
                mapping = job_key_mapping[job['jobId']]
                workunit_id = mapping['workunit_id']
                workunit = status[workunit_id]

                # Has something changed?
                workunit_has_changed = 0

                if(workunit['status'] != job['status']):
                    workunit['status'] = job['status']
                    workunit_has_changed = 1

                if('status_summary' in workunit):

                    for status_val in job['arrayProperties']['statusSummary']:
                        if status_val not in workunit['status_summary']:
                            workunit['status_summary'][status_val] = job['arrayProperties']['statusSummary'][status_val]
                            workunit_has_changed = 1
                        elif(workunit['status_summary'][status_val] != job['arrayProperties']['statusSummary'][status_val]):
                            workunit['status_summary'][status_val] = job['arrayProperties']['statusSummary'][status_val]
                            workunit_has_changed = 1
                else:
                    workunit['status_summary'] = job['arrayProperties']['statusSummary'].copy()
                    workunit_has_changed = 1


                if(workunit_has_changed == 1):
                    # status has changed
                    workunits_to_recompute[workunit_id] = 1

                    for subjob_id, subjob in workunit['subjobs'].items():
                        # Initialize subjob status if not present
                        if 'status' not in subjob:
                            subjob['status'] = 'PENDING'

                        if(subjob['status'] not in final_states):
                            subjobs_to_check.append({
                                    'workunit_id': workunit_id,
                                    'subjob_id': subjob_id,
                                    'batch_jobid': f"{job['jobId']}:{subjob_id}"
                                }
                            )
        else:
            logging.error(f"Did not get jobs response from AWS Batch")

    # Now let's check on the ones that have finished

    subjobs_to_parse = []

    print("Getting subjob status")

    for status_index_begin in range(0, len(subjobs_to_check), 100):
        job_keys_to_check = []
        job_key_mapping = {}

        for status_item in subjobs_to_check[status_index_begin:(status_index_begin+100)]:
            workunit_id = status_item['workunit_id']
            subjob_id = status_item['subjob_id']
            batch_jobid = status_item['batch_jobid']

            job_keys_to_check.append(batch_jobid)
            job_key_mapping[batch_jobid] = { 'workunit_id': workunit_id, 'subjob_id': subjob_id}

        response = client.describe_jobs(jobs=job_keys_to_check)

        if 'jobs' in response:
            for job in response['jobs']:
                mapping = job_key_mapping[job['jobId']]

                workunit_id = mapping['workunit_id']
                subjob_id = mapping['subjob_id']
                subjob = status[workunit_id]['subjobs'][subjob_id]

                subjob['status'] = job['status']

                if subjob['status'] in final_states:

                    if(subjob['status'] == "SUCCEEDED"):
                        subjobs_to_parse.append({ 'workunit_id': workunit_id, 'subjob_id': subjob_id})

                    subjob['vcpu_seconds_interrupted'] = 0
                    subjob['vcpu_seconds'] = 0

                    # Track timing metrics for analysis
                    if 'queue_wait_times' not in complete['summary']:
                        complete['summary']['queue_wait_times'] = []
                        complete['summary']['execution_times'] = []

                    # Calculate queue wait and execution time from job metadata
                    if 'createdAt' in job and 'startedAt' in job:
                        queue_wait = (job['startedAt'] - job['createdAt']) / 1000
                        complete['summary']['queue_wait_times'].append(queue_wait)

                    last_attempt_index = len(job['attempts']) - 1
                    for attempt_index, attempt in enumerate(job['attempts']):
                        if ( 'stoppedAt' in attempt and 'startedAt' in attempt):
                            vcpu_time = (attempt['stoppedAt'] - attempt['startedAt']) / 1000 * vcpus_per_job
                            execution_time = (attempt['stoppedAt'] - attempt['startedAt']) / 1000

                            if(attempt_index == last_attempt_index):
                                subjob['vcpu_seconds'] = vcpu_time
                                complete['summary']['execution_times'].append(execution_time)
                            else:
                                subjob['vcpu_seconds_interrupted'] += vcpu_time

                    complete['summary']['vcpu_seconds'] += subjob['vcpu_seconds']
                    complete['summary']['vcpu_seconds_interrupted'] += subjob['vcpu_seconds_interrupted']


                else:
                    pass
                    # Non-final state.


    # Now, we need to go check the files to see how many ligands there actually were

    download_queue = Queue()
    summary_queue = Queue()

    number_of_downloaders = multiprocessing.cpu_count() * 2
    downloader_processes = []
    for i in range(number_of_downloaders):
        downloader_processes.append(Process(target=downloader, args=(download_queue, summary_queue, config['temp_dir'], config)))
        downloader_processes[i].start()


    for item in subjobs_to_parse:
        workunit_id = item['workunit_id']
        subjob_id = item['subjob_id']

        # Get the link to the output
        # Use workunit_prefix for recovery jobs (will be "recovery-" or "")
        item['s3_path'] = f"{config['object_store_job_prefix']}/{config['job_name']}/summary/{workunit_prefix}{workunit_id}/{subjob_id}.json.gz"

        download_queue.put(item)

    for i in range(number_of_downloaders):
        download_queue.put(None)


    print("Downloading result files ")

    not_parsed = set((item['workunit_id'], item['subjob_id']) for item in subjobs_to_parse)

    download_procs_finished = 0
    download_count = 0
    while True:
        # Bounded wait: a downloader that died never sends its stop marker.
        # Liveness is read before the wait, so once all are dead an empty wait
        # means everything they sent has been received.
        downloaders_alive = any(proc.is_alive() for proc in downloader_processes)
        try:
            item = summary_queue.get(timeout=10)
        except Empty:
            if not downloaders_alive:
                logging.error(f"All summary downloaders exited before finishing")
                break
            continue

        # check for stop
        if item is None:
            download_procs_finished += 1
            if(download_procs_finished == number_of_downloaders):
                break
            continue


        download_count += 1

        # Print  flush=True
        if(download_count % 100 == 0):
            print(f".", flush=True, end="")


        # Otherwise process
        workunit_id = item['workunit_id']
        subjob_id = item['subjob_id']
        not_parsed.discard((workunit_id, subjob_id))

        subjob = status[workunit_id]['subjobs'][subjob_id]

        if item['result'] is None:
            # No usable summary: FAILED, so recovery reruns it, and no counts.
            # The workunit keeps the status Batch reported, so it is still
            # described; the report counts it FAILED once it is final.
            subjob['status'] = 'FAILED'
            workunits_to_recompute[workunit_id] = 1
            continue

        subjob['overview'] = {
           'total_dockings' : item['result']['total_dockings'],
           'docking_succeeded' : item['result']['dockings_status']['success'],
           'docking_failed' : item['result']['dockings_status']['failed'],
           'skipped_ligands' : item['result']['skipped_ligands'],
           'failed_downloads' : item['result']['failed_downloads'],
           'failed_downloads_dockings' : item['result']['failed_downloads_dockings']
        }

        for log_item in item['result']['failed_downloads_log']:
            failed_downloads.append([str(workunit_id), str(subjob_id), log_item['base_collection_key'], str(log_item['dockings']), log_item['reason']  ])

        for attr, attr_val in subjob['overview'].items():
            complete['summary'][attr] += attr_val

    for workunit_id, subjob_id in sorted(not_parsed):
        logging.error(f"No summary was received for {workunit_id}:{subjob_id}")
        status[workunit_id]['subjobs'][subjob_id]['status'] = 'FAILED'
        workunits_to_recompute[workunit_id] = 1

    print("")


    # submitted only workunits



    # re-calc workunits as needed

    for workunit_id in workunits_to_recompute:
        workunit = status[workunit_id]

        workunit['overview_status'] = {}
        for status_val in statuses:
            workunit['overview_status'][status_val] = {
                'ligands': 0,
                'workunits': 0,
                'subjobs': 0
            }

        # A final workunit with a failed subjob counts as FAILED, as Batch
        # fails an array with a failed child: a subjob failed closed above
        # (no usable summary) fails its workunit in the report, which the
        # wrapper starts recovery from
        workunit_status = workunit['status']
        if workunit_status in final_states and any(subjob['status'] == 'FAILED' for subjob in workunit['subjobs'].values()):
            workunit_status = 'FAILED'
        workunit['overview_status'][workunit_status]['workunits'] += 1

        for subjob_id, subjob in workunit['subjobs'].items():
            subjob_status = subjob['status']

            workunit['overview_status'][subjob_status]['subjobs'] += 1
            workunit['overview_status'][subjob_status]['ligands'] += subjob['ligands_expected']

    # now do new sum

    complete['overview_status'] = {}
    for status_val in statuses:
        complete['overview_status'][status_val] = {
            'ligands': 0,
            'workunits': 0,
            'subjobs': 0
        }

    for workunit_id, workunit in status.items():
        if 'status' not in workunit:
            # workunit has not been submitted yet
            continue

        # Initialize overview_status if it doesn't exist (for recovery jobs)
        if 'overview_status' not in workunit:
            workunit['overview_status'] = {}
            for status_val in statuses:
                workunit['overview_status'][status_val] = {
                    'ligands': 0,
                    'workunits': 0,
                    'subjobs': 0
                }

        for status_val in statuses:
            complete['overview_status'][status_val]['ligands'] += workunit['overview_status'][status_val]['ligands']
            complete['overview_status'][status_val]['workunits'] += workunit['overview_status'][status_val]['workunits']
            complete['overview_status'][status_val]['subjobs'] += workunit['overview_status'][status_val]['subjobs']


    # Output all failed downloads

    with open("../workflow/failed_downloads.csv", "a") as failed_downloads_out:
        for failed_download in failed_downloads:
            failed_downloads_out.write(",".join(failed_download))
            failed_downloads_out.write("\n")



    # output a nice summary

    status_header = "Status"
    workunits_header = "Workunits"
    subjobs_header = "Subjobs"
    ligands_header = "Dockings (est.)"

    print(f"-----------------------------------------------------------------")
    print("AWS Batch Progress")
    print(f"-----------------------------------------------------------------")
    print(f"")


    print(f"Docking count is inaccurate for prescreens. Correct value will")
    print(f"be in 'Completed Summary' when finished.")
    print(f"")

    print(f"{status_header:>15}   {workunits_header:^10}   {subjobs_header:^15}  {ligands_header:^15}")
    for status_val in statuses:
        print(f"{status_val:>15}   {complete['overview_status'][status_val]['workunits']:^10}   {complete['overview_status'][status_val]['subjobs']:^15}   {complete['overview_status'][status_val]['ligands']:^15}")


    vcpu_hrs = complete['summary']['vcpu_seconds'] / 60 / 60
    vcpu_hrs_interrupted = complete['summary']['vcpu_seconds_interrupted'] / 60 / 60

    print(f"")
    active_vcpus = (complete['overview_status']['RUNNING']['subjobs'] + complete['overview_status']['STARTING']['subjobs']) * vcpus_per_job
    print(f"Active vCPUs: {active_vcpus}")


    # Only show completed summary for main jobs (not recovery)
    if not is_recovery_mode:
        print(f"")
        print(f"-----------------------------------------------------------------")
        print("Completed Summary")
        print(f"-----------------------------------------------------------------")
        print(f"")
        print(f"* Total Dockings  : {complete['summary']['total_dockings']}")
        print(f"  - Succeeded     : {complete['summary']['docking_succeeded']}")
        print(f"  - Failed        : {complete['summary']['docking_failed']}")
        print(f"* Skipped ligands : {complete['summary']['skipped_ligands']}")
        print(f"* Failed Downloads: {complete['summary']['failed_downloads']} (est. {complete['summary']['failed_downloads_dockings']} dockings)")
        if(complete['summary']['failed_downloads'] != 0):
            print(f"     (failed downloads are in '../workflow/failed_downloads.csv')")
        print(f"")


        if(complete['summary']['total_dockings'] != 0):
            vcpu_sec = complete['summary']['vcpu_seconds'] / complete['summary']['total_dockings']
            print(f"* vCPU seconds per docking: {vcpu_sec:0.2f}")
        else:
            print(f"* vCPU seconds per docking: N/A")
        print(f"* vCPU hours total        : {vcpu_hrs:0.2f}")
        print(f"* vCPU hours interrupted  : {vcpu_hrs_interrupted:0.2f}")
        print(f"")

        # Add job timing analysis
        if 'queue_wait_times' in complete['summary'] and len(complete['summary']['queue_wait_times']) > 0:
            queue_times = complete['summary']['queue_wait_times']
            exec_times = complete['summary']['execution_times']

            print(f"-----------------------------------------------------------------")
            print(f"Job Timing Analysis ({len(queue_times)} completed jobs)")
            print(f"-----------------------------------------------------------------")
            print(f"* Avg queue wait time : {sum(queue_times)/len(queue_times):0.2f}s (min: {min(queue_times):0.2f}s, max: {max(queue_times):0.2f}s)")
            print(f"* Avg execution time  : {sum(exec_times)/len(exec_times):0.2f}s (min: {min(exec_times):0.2f}s, max: {max(exec_times):0.2f}s)")
            print(f"* Max execution time  : {max(exec_times):0.2f}s")

            # Detect if jobs are taking too long
            if max(exec_times) > 1200:  # 20 minutes
                print(f"")
                print(f"⚠️  WARNING: Some jobs taking >{max(exec_times)/60:.1f} min - likely S3 throttling or other bottleneck!")
                print(f"⚠️  Expected job time with {vcpus_per_job} vCPUs: ~{(vcpu_sec * 200 / vcpus_per_job):.1f}s for 200 ligands")

            print(f"")

    # Save the updated status to the appropriate file
    if is_recovery_mode:
        # Update the recovery_workunits in the recovery_complete structure
        recovery_complete['recovery_workunits'] = status
        with open("../workflow/status.recovery.tmp.json", "w") as json_out:
            json.dump(recovery_complete, json_out)
        shutil.move("../workflow/status.recovery.tmp.json", "../workflow/status.recovery.json")
    else:
        # Update the workunits in the complete structure
        complete['workunits'] = status
        with open("../workflow/status.tmp.json", "w") as json_out:
            json.dump(complete, json_out)
        shutil.move("../workflow/status.tmp.json", "../workflow/status.json")



def batch_job_vcpus(job, config):
    """vCPUs of a Batch job as Batch reports it (overrides included)."""
    container = job.get('container', {})
    vcpus = None
    for requirement in container.get('resourceRequirements', []):
        if requirement.get('type') == 'VCPU':
            vcpus = float(requirement['value'])
            break
    if vcpus is None and container.get('vcpus') is not None:
        vcpus = float(container['vcpus'])
    if vcpus is None:
        vcpus = float(config['aws_batch_subjob_vcpus'])
    return int(vcpus) if vcpus.is_integer() else vcpus


def describe_batch_jobs(client, job_ids):
    jobs = {}
    for status_index_begin in range(0, len(job_ids), 100):
        response = client.describe_jobs(jobs=job_ids[status_index_begin:(status_index_begin+100)])
        if 'jobs' in response:
            for job in response['jobs']:
                jobs[job['jobId']] = job
        else:
            logging.error(f"Did not get jobs response from AWS Batch")
    return jobs


def process_presigned(config):
    with afvs_presign.recovery_lock():
        process_presigned_locked(config)


def process_presigned_locked(config):
    """Status of a job_storage_mode=presigned screen. Reads nothing the docking
    jobs wrote: states and vCPU time come only from AWS Batch DescribeJobs (a
    source the jobs cannot alter). Docking counts are not reported here; the
    sealed tool container derives them from the summaries (progress.json)."""
    import boto3
    from botocore.config import Config

    statuses = ['SUBMITTED','PENDING','RUNNABLE','STARTING','RUNNING','SUCCEEDED','FAILED']
    final_states = ['SUCCEEDED', 'FAILED']

    parser = argparse.ArgumentParser()
    parser.add_argument('--detailed', action='store_true',
        help="Get detailed ligand information (this can take a long time!)")
    parser.add_argument('--recovery', action='store_true',
        help="Show only recovery jobs status")
    parser.parse_args()

    client = boto3.client('batch', config=Config(region_name=config['aws_region']))

    status_file = "../workflow/status.json"
    recovery_status_file = "../workflow/status.recovery.json"

    with open(status_file, "r") as read_file:
        complete = json.load(read_file)

    recovery_complete = None
    if os.path.exists(recovery_status_file):
        with open(recovery_status_file, "r") as read_file:
            recovery_complete = json.load(read_file)

    main_workunits = complete['workunits']
    # Rounds are appended in order; the sort (stable) only guards against reordering
    recovery_workunits = []
    if recovery_complete is not None:
        recovery_workunits = sorted(recovery_complete['recovery_workunits'].items(),
                                    key=lambda item: item[1].get('round', 0))

    # A recovery workunit recorded but not yet submitted (job_id None) has no
    # job to ask about (see afvs_presign.pending_in_grace for how it counts)
    submitted = [workunit for workunit in main_workunits.values() if workunit.get('job_id')] + \
        [workunit for _, workunit in recovery_workunits if workunit.get('job_id')]

    print("Getting workunit status")

    asked_parents = [workunit['job_id'] for workunit in submitted if workunit.get('status') not in final_states]
    parents = describe_batch_jobs(client, asked_parents)
    for workunit in submitted:
        job = parents.get(workunit['job_id'])
        if job is not None:
            workunit['status'] = job['status']
            workunit['status_summary'] = job.get('arrayProperties', {}).get('statusSummary', {}).copy()
        elif 'status' not in workunit:
            workunit['status'] = 'SUBMITTED'

    print("Getting subjob status")

    # Only the workunit's own subjob indexes are asked for, so the extra child
    # of a size-1 array (submitted as size 2) is never read
    subjobs_to_check = {}
    for workunit in submitted:
        for subjob_id, subjob in workunit['subjobs'].items():
            if 'status' not in subjob:
                subjob['status'] = 'SUBMITTED'
            if subjob['status'] not in final_states:
                subjobs_to_check[f"{workunit['job_id']}:{subjob_id}"] = (subjob, workunit)
        # A Batch array of one is submitted as two: the padding child does no
        # work but still runs, so its time is billed with the workunit
        if len(workunit['subjobs']) == 1 and '1' not in workunit['subjobs']:
            padding = workunit.setdefault('padding', {'status': 'SUBMITTED'})
            if padding['status'] not in final_states:
                subjobs_to_check[f"{workunit['job_id']}:1"] = (padding, workunit)

    children = describe_batch_jobs(client, list(subjobs_to_check))
    for batch_jobid, (subjob, workunit) in subjobs_to_check.items():
        job = children.get(batch_jobid)
        if job is None:
            # A child Batch never created (its array ended first) or no longer
            # reports (aged out with its parent) would otherwise stay
            # SUBMITTED for good and the screen would never finish: fail it
            # closed, unbilled, and recovery reruns it
            # DescribeJobs can lag a SubmitJob, so a job missing from it only
            # counts as gone once it was submitted a while ago
            parent_gone = (workunit['job_id'] in asked_parents and workunit['job_id'] not in parents
                           and time.time() - workunit.get('submitted_at', workunit.get('recorded_at', 0)) > PARENT_GRACE_SECONDS)
            if workunit.get('status') in final_states or parent_gone:
                subjob['status'] = 'FAILED'
                subjob['vcpu_seconds'] = 0
                subjob['vcpu_seconds_interrupted'] = 0
            continue

        subjob['status'] = job['status']
        subjob['vcpus'] = batch_job_vcpus(job, config)

        if subjob['status'] in final_states:
            attempts = job.get('attempts', [])
            subjob['vcpu_seconds'] = 0
            subjob['vcpu_seconds_interrupted'] = 0
            for attempt_index, attempt in enumerate(attempts):
                if('stoppedAt' in attempt and 'startedAt' in attempt):
                    vcpu_time = (attempt['stoppedAt'] - attempt['startedAt']) / 1000 * subjob['vcpus']
                    subjob['vcpu_seconds'] += vcpu_time
                    if(attempt_index != len(attempts) - 1):
                        subjob['vcpu_seconds_interrupted'] += vcpu_time

    # Latest state of each original subjob: the latest recovery round that
    # reran it, else its main entry
    latest = {}
    for workunit_id, workunit in main_workunits.items():
        if 'job_id' not in workunit:
            continue
        for subjob_id, subjob in workunit['subjobs'].items():
            latest[(workunit_id, subjob_id)] = subjob

    now = time.time()
    for _, recovery_workunit in recovery_workunits:
        pending = afvs_presign.pending_in_grace(recovery_workunit, now)
        if not recovery_workunit.get('job_id') and not pending:
            continue
        for recovery_subjob_id, mapping in recovery_workunit.get('recovery_subjobs_mapping', {}).items():
            key = (str(mapping['original_workunit_id']), str(mapping['original_subjob_id']))
            recovery_subjob = recovery_workunit['subjobs'].get(recovery_subjob_id)
            if key not in latest or recovery_subjob is None:
                continue
            if pending:
                # Its job may exist and not be listed yet: in flight, as
                # recovery counts it
                latest[key] = {**recovery_subjob, 'status': 'SUBMITTED'}
            elif 'status' in recovery_subjob:
                latest[key] = recovery_subjob

    overview_status = {}
    for status_val in statuses:
        overview_status[status_val] = {
            'ligands': 0,
            'workunits': 0,
            'subjobs': 0
        }

    active_vcpus = 0
    for workunit_id, workunit in main_workunits.items():
        if 'job_id' not in workunit:
            # workunit has not been submitted yet
            continue

        subjob_states = []
        for subjob_id, subjob in workunit['subjobs'].items():
            latest_subjob = latest[(workunit_id, subjob_id)]
            subjob_status = latest_subjob['status']
            subjob_states.append(subjob_status)

            overview_status[subjob_status]['subjobs'] += 1
            overview_status[subjob_status]['ligands'] += subjob['ligands_expected']

            if subjob_status in ('RUNNING', 'STARTING'):
                active_vcpus += latest_subjob.get('vcpus', int(config['aws_batch_subjob_vcpus']))

        if not subjob_states:
            continue
        if all(state == 'SUCCEEDED' for state in subjob_states):
            workunit_status = 'SUCCEEDED'
        else:
            workunit_status = 'FAILED'
            for state in ('RUNNING', 'STARTING', 'RUNNABLE', 'PENDING', 'SUBMITTED'):
                if state in subjob_states:
                    workunit_status = state
                    break
        overview_status[workunit_status]['workunits'] += 1

    # Billed vCPU time: every attempt of every subjob, main and all recovery rounds
    vcpu_seconds = 0
    vcpu_seconds_interrupted = 0
    for workunit in submitted:
        for subjob in list(workunit['subjobs'].values()) + [workunit.get('padding', {})]:
            vcpu_seconds += subjob.get('vcpu_seconds', 0)
            vcpu_seconds_interrupted += subjob.get('vcpu_seconds_interrupted', 0)

    complete['overview_status'] = overview_status
    complete['summary']['vcpu_seconds'] = vcpu_seconds
    complete['summary']['vcpu_seconds_interrupted'] = vcpu_seconds_interrupted


    # output a nice summary

    status_header = "Status"
    workunits_header = "Workunits"
    subjobs_header = "Subjobs"
    ligands_header = "Dockings (est.)"

    print(f"-----------------------------------------------------------------")
    print("AWS Batch Progress")
    print(f"-----------------------------------------------------------------")
    print(f"")

    print(f"{status_header:>15}   {workunits_header:^10}   {subjobs_header:^15}  {ligands_header:^15}")
    for status_val in statuses:
        print(f"{status_val:>15}   {overview_status[status_val]['workunits']:^10}   {overview_status[status_val]['subjobs']:^15}   {overview_status[status_val]['ligands']:^15}")


    vcpu_hrs = vcpu_seconds / 60 / 60
    vcpu_hrs_interrupted = vcpu_seconds_interrupted / 60 / 60

    print(f"")
    print(f"Active vCPUs: {active_vcpus}")

    print(f"")
    print(f"-----------------------------------------------------------------")
    print("Completed Summary")
    print(f"-----------------------------------------------------------------")
    print(f"")
    # The wrapper's awk patterns match whole lines, so this line must not
    # contain a state name or a docking-count label
    print(f"Docking counts are not read here; they come from the tool container's progress.json")
    print(f"")
    print(f"* vCPU seconds per docking: N/A")
    print(f"* vCPU hours total        : {vcpu_hrs:0.2f}")
    print(f"* vCPU hours interrupted  : {vcpu_hrs_interrupted:0.2f}")
    print(f"")

    with open("../workflow/status.tmp.json", "w") as json_out:
        json.dump(complete, json_out)
    shutil.move("../workflow/status.tmp.json", status_file)

    if recovery_complete is not None:
        with open("../workflow/status.recovery.tmp.json", "w") as json_out:
            json.dump(recovery_complete, json_out)
        shutil.move("../workflow/status.recovery.tmp.json", recovery_status_file)


def main():

    config = parse_config("../workflow/config.json")

    if(config['job_storage_mode'] == "s3" and config['batchsystem'] == "awsbatch"):
        with tempfile.TemporaryDirectory(prefix=os.path.join(config['tempdir_default'], '')) as temp_dir:
            config['temp_dir'] = temp_dir
            process(config)
    elif(config['job_storage_mode'] == "presigned" and config['batchsystem'] == "awsbatch"):
        process_presigned(config)
    else:
        print(f"Configuration with job_storage_mode={config['job_storage_mode']} and batchsystem={config['batchsystem']} is not supported")
        sys.exit(2)


if __name__ == '__main__':
    main()
