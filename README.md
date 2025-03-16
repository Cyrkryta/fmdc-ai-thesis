<div align="center">    

# Field Map Distortion Correction Master's Thesis for IT & Cognition

</div>

## TL;DR
Estimate and evaluate undistorted fMRI scans from a distorted scan and T1-weighted scan.

#### Description
The project aims to use a ```U-NET``` deep learning model for ```estimateing fieldmaps``` used for ```fMRI distortion correction```. It is a part of the ```final thesis``` for the IT & Cognition Master's program at the ```University of Copenhagen```, and is carried out in collaboration with the ```Neurobiology Research Unit (NRU)``` located at Rigshospitalet, Copenhagen. The thesis builds upon the work done by ```Jan Tagscherer```, a previous master's student affiliated with the ```NRU```.

## Data Acquisition and Processing
#### Data Acquisition
Identify datasets with data that should be used for testing and training. The respective dataset paths should be structured in ```trainval_file_paths.json``` and ```test_file_paths.json``` as follows, inserting the respective paths and subject holders.

    "ds004182": {
        "anat": "{subject_path}/anat/{subject}_rec-NORM_T1w.nii.gz",
        "fmap": {
            "magnitude": "{subject_path}/fmap/{subject}_magnitude1.nii.gz",
            "phasediff": "{subject_path}/fmap/{subject}_phasediff.nii.gz"
        },
        "func": "{subject_path}/func/{subject}_task-rest_run-1_bold.nii.gz",
        "echospacing": 0.000265001
    }
    
Once the files are created, used `datalad` to retrieve the `dataset` and `files`, and remove these again. First, retrieve datasets by running `datalad install https://github.com/OpenNeuroDatasets/{dataset}.git` manually. 

Retrieve the necessary files and move them to a seperate non-git folder by running the following commands for test and train (OBS! This assumes need for subsampling subjects, hence --n_subjects).

Training command:
```
python move_datalad_files.py --SOURCE_DIR_PATH /path/to/trainval_data_source --DEST_DIR_PATH /path/to/trainval_destination --CONFIG_PATH /path/to/trainval_file_paths.json --n_subjects {number of subjects to subsample} 
```

Test command:
```
python move_datalad_files.py --SOURCE_DIR_PATH /path/to/test_data_source --DEST_DIR_PATH /path/to/test_destination --CONFIG_PATH /path/to/test_file_paths.json --n_subjects {number of subjects to subsample} 
```

#### Data Preprocessing
To convert data to a suitable format for the model, the `prepare_data.py` file should be run. Once again, for overview reasons, create json files only following the format in the previous. 

Make sure to have `FSL` installed, and run the command

```
python prepare_data.py --FSL_DIR /path/to/fsl/root --SOURCE_DATASET_ROOT_DIR /path/to/retrieved/data/files/directory --DEST_DATASET_ROOT_DIR /path/to/processed/output/directory --JSON_PROCESSING_CONFIG_PATH /path/to/new/json/config/file
```

#### Training
Before training, it is important that you have the preprocessed data for it to work as intended. There are two ways of constructing the trian, validation and test data. `1.` Only add a training dataset. It will then be split 70/10/20. `2.` Add a training dataset and testing dataset, which will use the test dataset for testing and split the training data for train and validation (80/20). Run the following command to train a model

```
python project/training.py --TRAINING_DATASET_PATH "/path/to/processed/training/data/ds*" -- TEST_DATASET_PATH "/path/to/processed/test/data/ds*" --CHECKPOINT_PATH /path/to/checkpoint/folder --max-epochs 100000 --batch_size 32
```