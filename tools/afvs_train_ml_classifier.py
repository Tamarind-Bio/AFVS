#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trains the optional ML tranche-prioritization classifier (see tools/templates/ml_classifier.py) on
the results of an ATG Prescreen run.

Usage: run an ATG Prescreen as usual (prescreen_mode=1 in all.ctrl), then postprocess it with
tools/afvs_postprocess_atg-prescreen.sh (AWS/Athena) or
tools/afvs_postprocess_atg-prescreen.slurm.sh (Slurm/sharedfs). Then, from the prescreen job's own
tools/ directory, run this script once (a single host-side invocation -- no Slurm/AWS Batch job
needed, training only takes seconds):

    python3 afvs_train_ml_classifier.py --scenario-name <docking_scenario_name> --model-out <path>

This trains the classifier on the prescreen's per-ligand (SMILES, docking score) data and saves it.
The saved model is later consumed by tools/afvs_apply_ml_classifier_atg-primaryscreen.py to filter
the ATG Primary Screen's tranche-selected collections down to the molecules predicted worth
docking.

Two possible data sources are supported, matching the two postprocessing paths:
  - AWS/Athena: '<scenario>.ranking.complete.csv.gz' (produced by afvs_get_top_results.py via
    afvs_postprocess_atg-prescreen.sh), which retains every configured attr_* column including
    attr_smi.
  - Slurm/sharedfs: the raw per-collection summary files under
    'output-files/<scenario>/csv/**/*.csv.gz' (the same files
    afvs_postprocess_atg-prescreen.slurm.sh itself reads) -- there is no "complete" ranking file
    on this path, since it never goes through Athena.
Both sources are read by column NAME (not position) via pandas, since column order/count varies
with replica count and the print_attrs_in_summary setting in all.ctrl.

Requires PyTorch and RDKit installed on the host/login instance running this script (not required
inside the AWS Batch worker container, since this script never runs there).

@author: akshat
"""
import os
import sys
import glob
import argparse

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))
# ml_classifier.py lives in templates/ (not alongside this script) so that it is automatically
# bundled into the AWS Batch worker image (which ADDs templates/*.py), even though this training
# script itself is host-side only and is not bundled. This script therefore needs to add
# templates/ to sys.path explicitly to import it.
from ml_classifier import train_classifier  # noqa: E402


def load_ranking_dataframe(path):
    """
    Reads a gzip'd prescreen ranking/summary CSV and validates it has the columns needed for
    training (attr_smi, score_min), read by name (not position).

    Args:
        path (str): path to a gzip'd CSV file.

    Returns:
        pandas.DataFrame

    Raises:
        Exception: if attr_smi or score_min is missing from the file's columns.
    """
    df = pd.read_csv(path, compression='gzip')
    missing = [col for col in ('attr_smi', 'score_min') if col not in df.columns]
    if missing:
        raise Exception(
            'File {} is missing required column(s) {}. Ensure print_attrs_in_summary in all.ctrl '
            'includes "smi" for the prescreen run that produced this file.'.format(path, missing))
    return df


def load_training_data(scenario_name, source, ranking_complete_file, raw_csv_glob):
    """
    Loads (SMILES, docking score) training pairs from a prescreen run's ranking/summary data.

    Args:
        scenario_name (str): docking scenario name (used only for default path construction).
        source (str): 'aws', 'slurm', or 'auto' (try 'aws' first, fall back to 'slurm').
        ranking_complete_file (str): path to the AWS-path '<scenario>.ranking.complete.csv.gz' file.
        raw_csv_glob (str): glob pattern for the Slurm-path raw per-collection summary files.

    Returns:
        (smiles_list, scores_list)

    Raises:
        Exception: if the requested/auto-detected source has no usable data.
    """
    def _from_aws():
        if not os.path.isfile(ranking_complete_file):
            raise Exception('AWS ranking file not found: {}'.format(ranking_complete_file))
        df = load_ranking_dataframe(ranking_complete_file)
        df = df.dropna(subset=['attr_smi', 'score_min'])
        print('Loaded {} prescreen rows from {}'.format(len(df), ranking_complete_file))
        return df['attr_smi'].tolist(), df['score_min'].tolist()

    def _from_slurm():
        files = sorted(glob.glob(raw_csv_glob, recursive=True))
        if len(files) == 0:
            raise Exception('No raw per-collection summary files found matching: {}'.format(raw_csv_glob))
        frames = [load_ranking_dataframe(f) for f in files]
        df = pd.concat(frames, ignore_index=True)
        df = df.dropna(subset=['attr_smi', 'score_min'])
        print('Loaded {} prescreen rows from {} files matching {}'.format(
            len(df), len(files), raw_csv_glob))
        return df['attr_smi'].tolist(), df['score_min'].tolist()

    if source == 'aws':
        return _from_aws()
    if source == 'slurm':
        return _from_slurm()

    # auto: genuinely try the AWS "complete" ranking file first, falling back to raw Slurm csvs on
    # any failure (not just its absence -- e.g. it could exist but be missing attr_smi/score_min,
    # or be truncated/corrupt), rather than only pre-checking os.path.isfile().
    try:
        return _from_aws()
    except Exception as aws_error:
        try:
            return _from_slurm()
        except Exception as slurm_error:
            raise Exception(
                'No usable prescreen training data found. Tried AWS-path file {} ({}); '
                'tried Slurm-path glob {} ({}).'.format(
                    ranking_complete_file, aws_error, raw_csv_glob, slurm_error))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scenario-name', required=True,
                        help='Docking scenario name whose prescreen data to train on.')
    parser.add_argument('--source', choices=['auto', 'aws', 'slurm'], default='auto',
                        help='Which prescreen data source to read. Default: auto-detect.')
    parser.add_argument('--ranking-complete-file', default=None,
                        help='Path to the AWS-path ranking.complete.csv.gz file. '
                             'Default: ../output-files/<scenario-name>.ranking.complete.csv.gz')
    parser.add_argument('--raw-csv-glob', default=None,
                        help='Glob pattern for the Slurm-path raw per-collection summary files. '
                             'Default: ../output-files/<scenario-name>/csv/**/*.csv.gz')
    parser.add_argument('--model-out', required=True, help='Path to save the trained model (.pt file).')
    parser.add_argument('--top-percent', type=float, default=25,
                        help='Percentage of most-negative-scoring prescreen compounds labeled positive.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--receptor', default=None,
                        help='Optional receptor identifier stored as model metadata. '
                             'Defaults to --scenario-name if not given.')
    args = parser.parse_args()

    ranking_complete_file = args.ranking_complete_file or \
        '../output-files/{}.ranking.complete.csv.gz'.format(args.scenario_name)
    raw_csv_glob = args.raw_csv_glob or \
        '../output-files/{}/csv/**/*.csv.gz'.format(args.scenario_name)

    smiles_list, scores_list = load_training_data(
        args.scenario_name, args.source, ranking_complete_file, raw_csv_glob)

    if len(smiles_list) == 0:
        raise Exception('No training data loaded for scenario {}.'.format(args.scenario_name))

    model, history = train_classifier(
        smiles_list, scores_list, args.model_out,
        top_percent=args.top_percent, receptor=args.receptor or args.scenario_name, seed=args.seed)

    print('Classifier trained on {} prescreen compounds (best epoch {}, best val loss {:.4f}).'.format(
        len(smiles_list), history['best_epoch'] + 1, min(history['val_loss'])))
    print('Saved to: {}'.format(args.model_out))
    print('To use it for the ATG Primary Screen, pass --model {} to '
          'afvs_apply_ml_classifier_atg-primaryscreen.py.'.format(args.model_out))
