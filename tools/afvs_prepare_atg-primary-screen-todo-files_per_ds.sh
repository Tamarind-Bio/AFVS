#!/usr/bin/env bash
set -euo pipefail

usage="Usage: afvs_prepare_atg-primary-screen-todo-files.sh -m <tranche scoring method> [-r <tranche_filter_regex>] -s <size1>[:<size2>:...] <ds>

Description:
  Prepare Primary Screen todo files for ONE docking scenario (ds).
  Run this script once per ds (ds1, ds2, ...) and submit those jobs in parallel in Slurm.

Arguments:
  -m <tranche_scoring_mode>: dimension_averaging, tranche_min_score, tranche_ave_score
  -r <tranche_filter_regex>: Regex filter for tranche IDs (use '.*' for no filter)
  -s <size1>[:<size2>:...]: Number of ligands for Primary Screen (colon-separated)
  <ds>                    : docking scenario name, e.g. ds1
"

# Parse options
tranche_scoring_mode=""
tranche_filter_regex=""
sizes=()

while getopts "m:r:s:h" opt; do
    case $opt in
        m) tranche_scoring_mode=$OPTARG ;;
        r) tranche_filter_regex=$OPTARG ;;
        s) IFS=":" read -r -a sizes <<< "$OPTARG" ;;
        h) echo -e "\n${usage}\n" ; exit 0 ;;
        *) echo -e "\nInvalid option: -$OPTARG\n" ; echo -e "${usage}\n" ; exit 1 ;;
    esac
done

# Remaining positional args after getopts
shift $((OPTIND - 1))
ds="${1:-}"

# Checking arguments
if [[ -z "${tranche_scoring_mode}" ]]; then
    echo -e "\nError: Tranche scoring mode is required.\n"
    echo -e "${usage}\n"
    exit 1
elif [[ ${#sizes[@]} -eq 0 ]]; then
    echo -e "\nError: Screening sizes are required.\n"
    echo -e "${usage}\n"
    exit 1
elif [[ -z "${ds}" ]]; then
    echo -e "\nError: Docking scenario name (ds) is required as final argument.\n"
    echo -e "${usage}\n"
    exit 1
fi

if [[ -z "${tranche_filter_regex}" ]]; then
    tranche_filter_regex=".................."   # default 18-char tranche id
fi

echo "Docking scenario: ${ds}"
echo "Tranche scoring mode: ${tranche_scoring_mode}"
echo "Tranche filter regex: ${tranche_filter_regex}"
echo "Sizes: ${sizes[*]}"
echo

# Initial setup
cd ../output-files

# Input subset file must exist (generated from prescreen results)
# Postprocess script now produces a gzipped subset (.csv.gz) -- the downstream
# create_todofile_atg-primaryscreen.py uses compression='gzip' on this file.
subset_file="${ds}.ranking.subset-1.csv.gz"
if [[ ! -f "${subset_file}" ]]; then
  echo "ERROR: Missing ${subset_file} in ../output-files"
  exit 1
fi

# Getting the score averages for each tranche (only for this ds)
if [[ "${tranche_scoring_mode}" == "dimension_averaging" ]]; then
  inputfile="${subset_file}"
  echo "Generating dimension averaged tranche scores for ${ds} -> ${ds}.dimension-averaged-activity-map.csv ..."
  for i in {0..17}; do
    for a in {A..F}; do
      echo -n "${i},${a},"
      zgrep -E "^.{$i}$a" "${inputfile}" | awk -F ',' '{ total += $NF; count++ } END { if (count>0) print total/count; else print "" }' || echo
    done
  done | sed "1i\Tranche,Class,Score" | tee "${ds}.dimension-averaged-activity-map.csv"
fi

# Generating new todo files for THIS ds
COLL_PARQUET="${COLLECTIONS_PARQUET:-$HOME/Enamine_REAL_Space_2022q12.collections.parquet}"
TRAN_PARQUET="${TRANCHES_PARQUET:-$HOME/Enamine_REAL_Space_2022q12.tranches.parquet}"

if [[ "${tranche_scoring_mode}" == "dimension_averaging" ]]; then
  activity="${ds}.dimension-averaged-activity-map.csv"
  for size in "${sizes[@]}"; do
    echo "Generating todo for ${ds} size=${size} -> ${ds}.todo.${size}"
    echo "python3 ../tools/templates/create_todofile_atg-primaryscreen.py \
      "${activity}" "${COLL_PARQUET}" "${TRAN_PARQUET}" \
      "${tranche_scoring_mode}" "${ds}.todo.${size}" "${size}""
    python3 ../tools/templates/create_todofile_atg-primaryscreen.py \
      "${activity}" "${COLL_PARQUET}" "${TRAN_PARQUET}" \
      "${tranche_scoring_mode}" "${ds}.todo.${size}" "${size}"
  done

elif [[ "${tranche_scoring_mode}" == "tranche_min_score" || "${tranche_scoring_mode}" == "tranche_ave_score" ]]; then
  for size in "${sizes[@]}"; do
    echo "Generating todo for ${ds} size=${size} -> ${ds}.todo.${size}"

    echo "python3 ../tools/templates/create_todofile_atg-primaryscreen.py \
      "${subset_file}" "${COLL_PARQUET}" "${TRAN_PARQUET}" \
      "${tranche_scoring_mode}" "${ds}.todo.${size}" "${size}" "${tranche_filter_regex}""

    python3 ../tools/templates/create_todofile_atg-primaryscreen.py \
      "${subset_file}" "${COLL_PARQUET}" "${TRAN_PARQUET}" \
      "${tranche_scoring_mode}" "${ds}.todo.${size}" "${size}" "${tranche_filter_regex}"
  done
else
  echo "ERROR: Unknown tranche scoring mode: ${tranche_scoring_mode}"
  exit 1
fi

cd ../tools
echo "Done for ${ds}"
