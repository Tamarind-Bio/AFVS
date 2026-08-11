#!/bin/sh

#Checking the input arguments
usage="Usage: afvs_prepare_atg-primary-screen-folders.sh screening_sizes:<size 1>,<size 2>,... replica_counts:<replica count 1>,<replica count 2>,... <docking method> [ml_classifier_model:<path to trained .pt file>] [ml_classifier_cutoff:<probability cutoff, default 0.5>]

Description: Preparing the folders for the ATG Primary Screens folders. For each docking scenario, and each specified screening size and replica count, one ATG Primary Screen folder will be created. Before running this script, the todo files have to be prepared for each screening size.

Arguments:
    screening_sizes:<size 1>:<size 2>:...: Number of ligands that should be screened in the ATG Primary Screen. Multiple sizes can be spcified if multiple ATG Primary Screens are planned to be run with different screening sizes.
    replica_counts:<replica count 1>,<replica count 2>,... Replica counts that should be used in the ATG Primary Screen. Multiple counts can be spcified if multiple ATG Primary Screens are planned to be run with different replica counts.
    <docking method>: The docking program to be used.
    ml_classifier_model:<path> (optional): Path to a classifier trained by afvs_train_ml_classifier.py. When given, the ATG Primary Screen is prepared to dock only the molecules (within each tranche-selected collection) predicted worth docking by the classifier, via tools/afvs_apply_ml_classifier_atg-primaryscreen.py. When omitted, behavior is unchanged from today.
    ml_classifier_cutoff:<float> (optional): Minimum predicted probability (exclusive) to keep a ligand. Default 0.5. Only used if ml_classifier_model: is given.
"

# Checking arguments
if [ "${1}" == "-h" ]; then
   echo -e "\n${usage}\n\n"
   exit 0
fi
if [ "$#" -lt "3" ] || [ "$#" -gt "5" ]; then
   echo -e "\nWrong number of arguments. Three to five arguments are required."
   echo -e "\n${usage}\n\n"
   echo -e "Exiting..."
   exit 1
fi


# Extract the arguments
screening_sizes="$1"
replica_counts="$2"
docking_scenario_method="$3"
ml_classifier_model="$4"
ml_classifier_cutoff="$5"

# Checking argument contents
# First argument
if [[ "$screening_sizes" == screening_sizes:* ]]; then
  # Remove the "size:" prefix
  screening_sizes=${screening_sizes#screening_sizes:}

  # Convert the comma-separated sizes into an array
  IFS=',' read -r -a screening_sizes <<< "$screening_sizes"
else
  echo "Error: First argument does not start with 'screening_sizes:'. Exiting..."
  exit 1
fi
# Second argument
if [[ "$replica_counts" == replica_counts:* ]]; then
  # Remove the "size:" prefix
  replica_counts=${replica_counts#replica_counts:}

  # Convert the comma-separated sizes into an array
  IFS=',' read -r -a replica_counts <<< "$replica_counts"
else
  echo "Error: Second argument does not start with 'replica_counts:'. Exiting..."
  exit 1
fi

# Fourth argument (optional): ML tranche-prioritization classifier
use_ml_classifier=0
ml_classifier_model_path=""
ml_classifier_probability_cutoff="0.5"
if [ -n "${ml_classifier_model}" ]; then
  if [[ "$ml_classifier_model" == ml_classifier_model:* ]]; then
    ml_classifier_model_path=${ml_classifier_model#ml_classifier_model:}
    # Resolving to an absolute path now, before the "cd ../output-files" below changes the
    # working directory, since this script is otherwise invoked from the tools/ directory.
    ml_classifier_model_path=$(realpath "${ml_classifier_model_path}")
    if [ -z "${ml_classifier_model_path}" ]; then
      echo "Error: Could not resolve ml_classifier_model: path with realpath. Check that the file exists. Exiting..."
      exit 1
    fi
    use_ml_classifier=1
  else
    echo "Error: Fourth argument does not start with 'ml_classifier_model:'. Exiting..."
    exit 1
  fi
fi

# Fifth argument (optional): ML classifier probability cutoff
if [ -n "${ml_classifier_cutoff}" ]; then
  if [[ "$ml_classifier_cutoff" == ml_classifier_cutoff:* ]]; then
    if [ "${use_ml_classifier}" -eq "0" ]; then
      echo "Warning: ml_classifier_cutoff: was given without ml_classifier_model:, so it has no effect (the ML classifier is not enabled)."
    fi
    ml_classifier_probability_cutoff=${ml_classifier_cutoff#ml_classifier_cutoff:}
  else
    echo "Error: Fifth argument does not start with 'ml_classifier_cutoff:'. Exiting..."
    exit 1
  fi
fi

# Initial setup
cd ../output-files
echo

# Prepare next stage foldersparent_dir=$(basename $(dirname $(pwd)))
for ds in $(cat ../workflow/config.json  | jq -r ".docking_scenario_names" | tr "," " " | tr -d '"\n[]' | tr -s " "); do
  for screening_size in "${screening_sizes[@]}"; do

    # Optional ML tranche-prioritization classifier: run once per (ds, screening_size) pair here
    # (not once per replica count below -- the ML-selected ligand list is independent of
    # docking_scenario_replicas), before the replica-count loop that creates the actual run
    # folders. Skipped if the output already exists, so re-running this script after a partial
    # failure does not redundantly re-fetch/re-score potentially millions of collections.
    ml_selected_file=""
    if [ "${use_ml_classifier}" -eq "1" ]; then
      ml_selected_file="../output-files/${ds}.todo.${screening_size}.ml-selected.csv"
      if [ -f "${ml_selected_file}" ]; then
        echo "ML-filtered ligand list already exists (${ml_selected_file}), skipping re-generation."
      else
        echo "Applying the ML tranche-prioritization classifier for docking scenario ${ds}, screening size ${screening_size}..."
        # ml_classifier_model_path is the one variable in this script resolved via realpath, so
        # (unlike the other, config-derived tokens used elsewhere here) it can contain spaces --
        # quoting it (and the other args, as cheap insurance) avoids word-splitting it into
        # multiple argv tokens.
        python3 ../tools/afvs_apply_ml_classifier_atg-primaryscreen.py --todo-file "../output-files/${ds}.todo.${screening_size}" --model "${ml_classifier_model_path}" --output "${ml_selected_file}" --probability-cutoff "${ml_classifier_probability_cutoff}"
      fi
    fi

    for replica_count in "${replica_counts[@]}"; do
      new_af_root_folder="../../atg-primaryscreen_${ds}_${screening_size}_repl${replica_count}"
      echo "Creating new AF directory (${new_af_root_folder}) for the ATG Primary Screen of docking scenrio ${ds} for screening size ${screening_size}, and ${replica_count} replicas"
      mkdir ${new_af_root_folder}
      mkdir ${new_af_root_folder}/input-files/
      cp -r ../.git* ../tools/ ${new_af_root_folder}/
      cp -r ../input-files/receptors/ ${new_af_root_folder}/input-files/
      cp -r ../input-files/${ds}/ ${new_af_root_folder}/input-files/
      echo "Copying the newly created todo file for docking scenario ${ds}: cp ../output-files/${ds}.todo.${screening_size} ${new_af_root_folder}/tools/templates/todo.all"
      cp ../output-files/${ds}.todo.${screening_size} ${new_af_root_folder}/tools/templates/todo.all
      echo "Setting job_name in the all.ctrl file to atg-primaryscreen_${ds}_${screening_size}_repl${replica_count}"
      sed -i "s/job_name=.*/job_name=atg-primaryscreen_${ds}_${screening_size}_repl${replica_count}/g" ${new_af_root_folder}/tools/templates/all.ctrl
      echo "Setting data_collection_identifier in the all.ctrl file to Enamine_REAL_Space_2022q12"
      sed -i "s|data_collection_identifier=.*|data_collection_identifier=Enamine_REAL_Space_2022q12|g" ${new_af_root_folder}/tools/templates/all.ctrl
      echo "Setting prescreen_mode in the all.ctrl file to 0"
      sed -i "s|prescreen_mode=.*|prescreen_mode=0|g" ${new_af_root_folder}/tools/templates/all.ctrl
      echo "Setting dynamic_tranche_filtering in the all.ctrl file to 0"
      sed -i "s|dynamic_tranche_filtering=.*|dynamic_tranche_filtering=0|g" ${new_af_root_folder}/tools/templates/all.ctrl
      echo "Setting docking_scenario_names in the all.ctrl file to ${ds}"
      sed -i "s|docking_scenario_names=.*|docking_scenario_names=${ds}|g" ${new_af_root_folder}/tools/templates/all.ctrl
      echo "Setting docking_scenario_batchsizes in the all.ctrl file to 1"
      sed -i "s|docking_scenario_batchsizes=.*|docking_scenario_batchsizes=1|g" ${new_af_root_folder}/tools/templates/all.ctrl
      echo "Setting docking_scenario_replicas in the all.ctrl file to ${replica_count}"
      sed -i "s|docking_scenario_replicas=.*|docking_scenario_replicas=${replica_count}|g" ${new_af_root_folder}/tools/templates/all.ctrl
      echo "Setting docking_scenario_methods in the all.ctrl file to ${docking_scenario_method}"
      sed -i "s|docking_scenario_methods=.*|docking_scenario_methods=${docking_scenario_method}|g" ${new_af_root_folder}/tools/templates/all.ctrl

      if [ "${use_ml_classifier}" -eq "1" ]; then
        echo "Using the ML-filtered ligand list for this ATG Primary Screen (collection_list_type=csv_collection_key_ligand)"
        cp ${ml_selected_file} ${new_af_root_folder}/tools/templates/atg-ml-selected-ligands.csv
        # Anchored to the start of the line (unlike the sed substitutions above, which also match
        # this key inside unrelated comments elsewhere in all.ctrl -- harmless since parse_config()
        # only recognizes lines starting with the key itself, but there is no reason for these two
        # new lines to introduce that same avoidable imprecision).
        sed -i "/^collection_list_type=/s|.*|collection_list_type=csv_collection_key_ligand|g" ${new_af_root_folder}/tools/templates/all.ctrl
        sed -i "/^collection_list_file=/s|.*|collection_list_file=templates/atg-ml-selected-ligands.csv|g" ${new_af_root_folder}/tools/templates/all.ctrl
      fi

      echo
    done
  done
done

# Changing to original directory
echo
cd ../tools

#for ds in $(cat ../workflow/config.json  | jq -r ".docking_scenario_names" | tr "," " " | tr -d '"\n[]' | tr -s " "); do for size in ${@:2}; do ( cd ../../atg-primaryscreen_${size}_${ds}/tools; ./afvs_prepare_folders.py ) ;  done; done
#for ds in $(cat ../workflow/config.json  | jq -r ".docking_scenario_names" | tr "," " " | tr -d '"\n[]' | tr -s " "); do for size in ${@:2}; do echo "( cd ../../atg-primaryscreen_${size}_${ds}/tools; ./afvs_prepare_workunits.py )" ; done; done | parallel -j 10 --ungroup
#for ds in $(cat ../workflow/config.json  | jq -r ".docking_scenario_names" | tr "," " " | tr -d '"\n[]' | tr -s " "); do for size in ${@:2}; do ( cd ../../atg-primaryscreen_${size}_${ds}/tools; ./afvs_build_docker.sh ) ; done; done
#for ds in $(cat ../workflow/config.json  | jq -r ".docking_scenario_names" | tr "," " " | tr -d '"\n[]' | tr -s " "); do for size in ${@:2}; do ( cd ../../atg-primaryscreen_${size}_${ds}/tools; ./afvs_submit_jobs.py 1 500 ) ; done; done

