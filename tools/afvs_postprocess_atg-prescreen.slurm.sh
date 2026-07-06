#!/bin/sh

#Checking the input arguments
usage="Usage: afvs_postprocess_atg-prescreen.sh

Description: Postprocessing the ATG Prescreen, and preparing the todo files for the ATG Primary Screens. The script can handle multiple docking scenarios in the ATG Prescreen.

"

if [ "${1}" == "-h" ]; then
   echo -e "\n${usage}\n\n"
   exit 0
fi
if [ "$#" -ne "0" ]; then
   echo -e "\nWrong number of arguments."
   echo -e "\n${usage}\n\n"
   echo -e "Exiting..."
   exit 1
fi

# Output-files directory
echo "Creating output-files folder (if it does not yet exist) ..."
mkdir -p ../output-files
cd ../output-files

# Creating subset of the CSV files (one gzipped output per docking scenario)
# Fixes vs. CG_1228 reference:
#   - actually pigz-compress the output (was uncompressed plain CSV)
#   - prepend the header row that downstream pandas/Python expects
#   - use > (overwrite) instead of >> (append), so reruns are idempotent
#   - process docking scenarios in parallel (background + wait) to use both cores
for ds in $(cat ../workflow/config.json  | jq -r ".docking_scenario_names" | tr "," " " | tr -d '"\n[]' | tr -s " "); do
  (
    out="../output-files/${ds}.ranking.subset-1.csv.gz"
    echo "[$(date +%H:%M:%S)] [${ds}] -> ${out}"
    {
      echo "Tranche,Collection,LigandAFID,ScoreMin"
      find ../output-files/${ds}/csv -name "*.csv.gz" -print0 \
        | xargs -0 pigz -dc \
        | awk -F, '$5 ~ /^-?[0-9]/ { split($2, a, "_"); print a[1] "," a[2] "," $1 "," $5 }'
    } | pigz -c > "${out}"
    echo "[$(date +%H:%M:%S)] [${ds}] done -> ${out} ($(stat -c %s "${out}") bytes)"
  ) &
done

wait

cd ../tools
