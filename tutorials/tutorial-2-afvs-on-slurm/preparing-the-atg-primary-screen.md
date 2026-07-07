# Preparing the ATG Primary Screen

After the ATG Prescreen has been postprocessed, we can prepare the ATG Primary Screen:&#x20;

* **AFVS Folder:** Preparation of a new AFVS folder for the ATG Primary Screen
* **Input Files:** The AFVS Folder needs to be prepared by copying the docking input files into it, as well as the todo file that contains the ligand collections that should be screened. In addition, the all.ctrl file has to be configured with the proper settings. &#x20;

All of the above can be automatically done with the following command to be run in the `tools` folder:

```
./afvs_prepare_atg-primary-screen.sh gk-vs1 10000
```

The first argument is a prefix that can be freely chosen, that used for the file path when the screening data is stored in S3. We chose the prefix gk-vs1, "gk" standing for the target glucokinase, and "vs1" for virtual screening 1. The screening size specified here has to match with the screening size that was specified during the postprocessing of the ATG Prescreen.

For each docking scenario and each specified screening size, a new AFVS folder was created in the parent folder `../../` (two levels above the current working directory), one for each ATG Primary Screen. In our case, we have one docking scenario called `gk_ds1`. We now change the directory to the corresponding ATG primary screen folder:

```
cd ../../atg-primary_10000_gk-ds1/tools
```

Then we prepare the workflow in the same way as we did for the ATG Prescreen:

```
./afvs_prepare_folders.py
./afvs_prepare_workunits.py
./afvs_build_docker.sh
```

The command ./afvs\_prepare\_workunits.py will print the total number of work units. Please remember this number, as it is needed in the next section.&#x20;

## Optional: ML Tranche-Prioritization Classifier

Instead of docking every ligand of every tranche-selected collection, you can optionally train a machine learning classifier on the ATG Prescreen's docking results and use it to keep, within each selected collection, only the ligands it predicts are worth docking. This can substantially reduce the number of ligands actually docked in the ATG Primary Screen.

First, train the classifier on the postprocessed ATG Prescreen results. This is a one-time, host-side command that only takes seconds to run (no Slurm job needed), to be run in the `tools` folder of the ATG Prescreen (not the new Primary Screen folder created above):

```
./afvs_train_ml_classifier.py --scenario-name gk_ds1 --source slurm --model-out atg-ml-classifier.pt
```

Unlike the AWS/Athena postprocessing path, the Slurm path has no single combined "ranking.complete" file, so `--source slurm` reads the raw per-collection summary files directly (`../output-files/<docking scenario name>/csv/**/*.csv.gz`, the same files `afvs_postprocess_atg-prescreen.sh` itself reads).

Then pass the trained model to `afvs_prepare_atg-primary-screen-folders.sh` as an additional argument, before creating the Primary Screen folders:

```
./afvs_prepare_atg-primary-screen-folders.sh screening_sizes:10000 replica_counts:1 qvina02 ml_classifier_model:atg-ml-classifier.pt
```

This scores every ligand of every tranche-selected collection (not just the sparse sample docked during the ATG Prescreen) and keeps only the ones predicted to have a binding probability greater than 0.5. This cutoff can be adjusted with an additional `ml_classifier_cutoff:<value>` argument. The new Primary Screen folder's `all.ctrl` is automatically configured to dock only the ML-selected ligands (listed in `tools/templates/atg-ml-selected-ligands.csv`) rather than the full tranche-selected todo file that is otherwise used.

Training and applying the classifier both require PyTorch and RDKit installed on this host/login instance. Run `./afvs_train_ml_classifier.py --help` or `./afvs_apply_ml_classifier_atg-primaryscreen.py --help` for the full list of options.

