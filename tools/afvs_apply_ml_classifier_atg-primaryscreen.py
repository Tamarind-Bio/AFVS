#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filters an ATG Primary Screen's tranche-selected collections down to the molecules predicted worth
docking, using a classifier trained by tools/afvs_train_ml_classifier.py.

Run this after tools/afvs_prepare_atg-primary-screen-todo-files.sh (or its _per_ds sibling) has
produced a tranche-selected todo file ('<scenario>.todo.<size>', "TrancheCollection LigandCount"
format), and before tools/afvs_prepare_atg-primary-screen-folders.sh sets up the primary-screen run
folder. Usage (from the prescreen job's own tools/ directory):

    python3 afvs_apply_ml_classifier_atg-primaryscreen.py \\
        --todo-file ../output-files/<scenario>.todo.<size> \\
        --model <path to trained .pt file> \\
        --output ../output-files/<scenario>.todo.<size>.ml-selected.csv

For every collection listed in --todo-file, this fetches the collection's FULL ligand set (not
just the sparse prescreen sample -- every ligand of every tranche-selected collection is scored),
extracts each ligand's SMILES, and scores all of a collection's ligands in a single batched call.
The output is a CSV in the 'csv_collection_key_ligand' format that all.ctrl's collection_list_type
already supports (header 'collection_name,collection_number,ligand'), so it can be dropped
straight into a new all.ctrl's collection_list_file with zero changes needed to afvs_run.py.

Collection fetching reuses this repo's own established mechanisms directly (imported, not
reimplemented): gen_s3_download_path()/gen_sharedfs_path() from afvs_prepare_workunits.py resolve
a collection's location under either addressing mode (hash/metatranche) and either storage backend
(s3/sharedfs); unpack_item()/get_attrs() from tools/templates/afvs_run.py extract a collection's
ligand files and their SMILES, exactly as the docking engine itself does.

Requires PyTorch and RDKit installed on the host/login instance running this script (not required
inside the AWS Batch worker container, since this script never runs there).

@author: akshat
"""
import os
import re
import sys
import csv
import json
import time
import uuid
import shutil
import argparse
import tempfile

import boto3
import botocore
from botocore.config import Config

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))
# ml_classifier.py and afvs_run.py both live in templates/ -- ml_classifier.py so it is auto-bundled
# into the AWS Batch worker image (which ADDs templates/*.py), afvs_run.py so its unpack_item()/
# get_attrs() helpers (the exact logic the docking engine itself uses) can be reused here without
# duplication. This script itself is host-side only and is not Docker-bundled.
from ml_classifier import load_classifier, filter_smiles_mask  # noqa: E402
from afvs_run import unpack_item, get_attrs  # noqa: E402

# afvs_prepare_workunits.py lives alongside this script in tools/, importable directly.
from afvs_prepare_workunits import gen_s3_download_path, gen_sharedfs_path  # noqa: E402


COLLECTION_KEY_RE = re.compile(r'(?P<collection_start>.*)_(?P<collection_number>\w+)$')
MIN_FREE_BYTES = 1024 * 1024 * 1024  # 1 GiB, matching afvs_run.py's own startup free-space check.


def read_workflow_config(path):
    """Reads the already-materialized workflow/config.json (not all.ctrl -- see module docstring
    in ml_classifier.py and the plan notes: gen_s3_download_path()/gen_sharedfs_path() require
    config fields, e.g. sharedfs_collection_path, that only exist after afvs_prepare_folders.py's
    full processing, which config.json already reflects)."""
    with open(path, 'r') as f:
        return json.load(f)


def read_todo_file(path):
    """
    Reads a tranche-selected todo file ('TrancheCollection LigandCount' per line, matching
    afvs_prepare_folders.py's own 'standard' collection_list_type parsing convention).

    Returns:
        list of (collection_name, collection_number) tuples, in file order. collection_number is
        kept as the exact string from the file (never reformatted/round-tripped through int()),
        since it must match the tar archive's internal zero-padded directory name exactly.
    """
    collections = []
    with open(path, 'r') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 1:
                print('WARNING: Skipping malformed todo-file line {}:{}: {}'.format(path, line_number, line))
                continue
            collection_key = parts[0]
            match = COLLECTION_KEY_RE.search(collection_key)
            if not match:
                print('WARNING: Could not parse collection key on line {}:{}: {}'.format(
                    path, line_number, collection_key))
                continue
            collections.append((match.group('collection_start'), match.group('collection_number')))
    return collections


def fetch_collection(config, collection_name, collection_number, tmp_dir):
    """
    Downloads/copies one collection's tarball and extracts every ligand in it, reusing the exact
    mechanisms afvs_run.py's own downloader()/untar() pipeline stages use.

    Returns:
        (ligands_dict_or_None, temp_dir): ligands_dict is None if the fetch or unpack failed
        (logged, caller should skip this collection); temp_dir is always created and must be
        cleaned up by the caller regardless of success.
    """
    ctx = {'config': config}
    temp_dir = tempfile.mkdtemp(dir=tmp_dir)
    local_path = os.path.join(temp_dir, 'tmp.tar.gz')

    try:
        collection = {
            'collection_name': collection_name,
            'collection_number': collection_number,
            'mode': 'all',  # matches afvs_prepare_folders.py's own literal value for "every ligand"
        }

        if config['data_storage_mode'] == 's3':
            s3_bucket = config['object_store_data_bucket']
            s3_download_path = gen_s3_download_path(ctx, collection_name, collection_number)
            botoconfig = Config(retries={'max_attempts': 25, 'mode': 'standard'})
            s3 = boto3.client('s3', config=botoconfig)
            try:
                with open(local_path, 'wb') as f:
                    s3.download_fileobj(s3_bucket, s3_download_path, f)
            except botocore.exceptions.ClientError as error:
                print('WARNING: Failed to download {}/{} from S3 ({}). Skipping collection {}_{}.'.format(
                    s3_bucket, s3_download_path, error, collection_name, collection_number))
                return None, temp_dir
        elif config['data_storage_mode'] == 'sharedfs':
            sharedfs_path = gen_sharedfs_path(ctx, collection_name, collection_number)
            try:
                shutil.copyfile(sharedfs_path, local_path)
            except OSError as error:
                print('WARNING: Failed to copy {} ({}). Skipping collection {}_{}.'.format(
                    sharedfs_path, error, collection_name, collection_number))
                return None, temp_dir
        else:
            raise Exception('data_storage_mode must be either s3 or sharedfs (currently: {})'.format(
                config['data_storage_mode']))

        item = {
            'temp_dir': temp_dir,
            'local_path': local_path,
            'collection_key': '{}_{}'.format(collection_name, collection_number),
            'collection': collection,
        }
        ligands = unpack_item(item)  # side effect: os.chdir(temp_dir) -- see module docstring
        if ligands is None:
            print('WARNING: Could not unpack collection {}_{}. Skipping.'.format(
                collection_name, collection_number))
            return None, temp_dir

        return ligands, temp_dir
    except Exception:
        # The caller only learns temp_dir's path via this function's return value, so if we're
        # about to propagate an exception instead of returning normally (e.g. a botocore error
        # that isn't a ClientError subclass, such as EndpointConnectionError), we must clean up
        # our own temp_dir here -- the caller has no way to do it for us.
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def check_disk_space(tmp_dir):
    stat = shutil.disk_usage(tmp_dir if tmp_dir else tempfile.gettempdir())
    if stat.free < MIN_FREE_BYTES:
        raise RuntimeError('Less than 1GB of space free in tmp dir ({}, free: {} bytes). Aborting.'.format(
            tmp_dir, stat.free))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--todo-file', required=True,
                        help='Tranche-selected todo file, e.g. ../output-files/<scenario>.todo.<size>')
    parser.add_argument('--model', required=True, help='Trained classifier .pt file.')
    parser.add_argument('--workflow-config', default='../workflow/config.json',
                        help='Path to the prescreen run\'s workflow/config.json.')
    parser.add_argument('--output', required=True, help='Destination csv_collection_key_ligand CSV.')
    parser.add_argument('--probability-cutoff', type=float, default=0.5,
                        help='Minimum predicted probability (exclusive) to keep a ligand.')
    parser.add_argument('--tmp-dir', default=None,
                        help='Base directory for per-collection temp extraction (default: system temp).')
    parser.add_argument('--progress-interval', type=int, default=100,
                        help='Print progress / check free disk space every N collections.')
    args = parser.parse_args()
    if args.progress_interval < 1:
        parser.error('--progress-interval must be a positive integer.')

    # Resolve every script-owned path to absolute *before* the collection loop: unpack_item()
    # calls os.chdir() as a side effect, so any relative path referenced later would silently
    # break after the first collection.
    todo_file = os.path.abspath(args.todo_file)
    model_path = os.path.abspath(args.model)
    workflow_config_path = os.path.abspath(args.workflow_config)
    output_path = os.path.abspath(args.output)
    tmp_dir = os.path.abspath(args.tmp_dir) if args.tmp_dir else None

    config = read_workflow_config(workflow_config_path)
    model, metadata = load_classifier(model_path)
    fp_radius = metadata.get('fp_radius', 2)
    fp_nbits = metadata.get('fp_nbits', 1024)

    collections = read_todo_file(todo_file)
    total_collections = len(collections)
    print('Loaded {} collections from {}'.format(total_collections, todo_file))

    check_disk_space(tmp_dir)

    output_dir = os.path.dirname(output_path)
    if output_dir != '' and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    tmp_output_path = '{}.tmp-{}'.format(output_path, uuid.uuid4().hex)

    n_kept_total = 0
    n_scored_total = 0
    n_collections_failed = 0
    original_cwd = os.getcwd()

    try:
        with open(tmp_output_path, 'w', newline='') as out_f:
            writer = csv.writer(out_f)
            writer.writerow(['collection_name', 'collection_number', 'ligand'])

            for counter, (collection_name, collection_number) in enumerate(collections, start=1):
                collection_temp_dir = None
                try:
                    # fetch_collection() itself (not just its return value) must be inside this
                    # try/except: an unhandled exception here (e.g. a transient network error not
                    # caught by fetch_collection()'s own narrower except clauses) must not abort
                    # the whole run -- with potentially thousands of collections and a long
                    # wall-clock runtime, one bad collection should be logged and skipped, not
                    # discard all progress on every other collection.
                    ligands, collection_temp_dir = fetch_collection(config, collection_name, collection_number, tmp_dir)
                    if ligands is None:
                        n_collections_failed += 1
                        continue

                    ligand_names = list(ligands.keys())
                    smiles_list = []
                    for ligand_name in ligand_names:
                        attrs = get_attrs(config['ligand_library_format'], ligands[ligand_name]['path'], attrs=['smi'])
                        smi = attrs.get('smi', 'N/A')
                        smiles_list.append('' if smi == 'N/A' else smi)

                    keep_mask, _ = filter_smiles_mask(
                        model, smiles_list, probability_cutoff=args.probability_cutoff,
                        fp_radius=fp_radius, fp_nbits=fp_nbits)

                    for ligand_name, keep in zip(ligand_names, keep_mask):
                        if keep:
                            writer.writerow([collection_name, collection_number, ligand_name])
                            n_kept_total += 1
                    n_scored_total += len(ligand_names)
                except Exception as e:
                    n_collections_failed += 1
                    print('WARNING: Unexpected error processing collection {}_{} ({}). Skipping.'.format(
                        collection_name, collection_number, e))
                finally:
                    # unpack_item() (called inside fetch_collection()) chdir()s into
                    # collection_temp_dir as a side effect; restore cwd before removing that
                    # directory (which would otherwise leave the process's cwd pointing at a
                    # just-deleted directory).
                    os.chdir(original_cwd)
                    if collection_temp_dir is not None:
                        shutil.rmtree(collection_temp_dir, ignore_errors=True)

                if counter % args.progress_interval == 0:
                    percent_done = (counter / total_collections) * 100 if total_collections > 0 else 100
                    print('{:.2f}% completed.... ({}/{} collections, {} ligands kept of {} scored, '
                          '{} collections failed)'.format(
                              percent_done, counter, total_collections, n_kept_total, n_scored_total,
                              n_collections_failed))
                    check_disk_space(tmp_dir)

        os.replace(tmp_output_path, output_path)
    except BaseException:
        # Covers both the loop body above and the "with open(...)"/os.replace() calls themselves:
        # don't leave a partially-written .tmp-<uuid> file behind on a hard failure (disk-full,
        # Ctrl-C, the disk-space check's RuntimeError, etc.).
        if os.path.exists(tmp_output_path):
            os.remove(tmp_output_path)
        raise

    print('Done. Scored {} ligands across {} collections ({} collections failed to fetch/unpack); '
          'kept {} above probability_cutoff={}.'.format(
              n_scored_total, total_collections, n_collections_failed, n_kept_total, args.probability_cutoff))
    print('Output written to: {}'.format(output_path))
