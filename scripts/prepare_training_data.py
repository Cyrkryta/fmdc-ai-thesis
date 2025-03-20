# Importing the necessary dependencies
import json
import os
import argparse
from os.path import abspath
import shutil
from glob import glob
import pandas as pd
from tqdm import tqdm
import nibabel as nib
import numpy as np
from nipype import Node, Workflow, Function
import nipype.interfaces.io as nio
from nipype.interfaces import fsl
from nipype.interfaces.fsl import BET, MeanImage, EpiReg, FLIRT, PrepareFieldmap, FUGUE, ConvertXFM
from nipype.interfaces.fsl.maths import ErodeImage
from nipype import SelectFiles

"""
Function for getting the median time frame
"""
def GetMedianTF(in_file):
    # Import necessary dependency
    import nibabel as nib
    # Return the median
    return int(nib.load(in_file).header['dim'][4] / 2)

"""
Function subtracting 5 from an input value
"""
def SubtractFive(in_value):
    return in_value - 5

"""
Intensity normalize an input file
"""
def IntensityNormalization(in_file):
    # Import the necessary dependencies
    import nibabel as nib
    import numpy as np
    # Load image and image data from the file
    img = nib.load(in_file)
    data = img.get_fdata()
    # Create intensity histogram, and define its cumulative representation
    image_histogram, bins = np.histogram(data[data != 0].flatten(), 256, density=True)
    cdf = image_histogram.cumsum()
    cdf = (256-1) * cdf / cdf[-1]
    # Perform interpolation of the image
    image_equalized = np.interp(data.flatten(), bins[:-1], cdf).reshape(data.shape)
    # Defining the output path and loading the new image
    out_path = f'{in_file}.norm.nii.gz'
    new_img = nib.Nifti1Image(image_equalized, img.affine)
    nib.save(new_img, out_path)
    # Returning the output path to the image
    return out_path


"""
Given a particular subject in a dataset, preprocess the data by following the pipeline.
"""
def _convert_subject(SUBJECT_INPUT_PATH: str, dataset, FSL_DIR: str):
    # Retrieve the full normalized path, retrieve the subject
    SUBJECT_NORM_PATH = os.path.normpath(SUBJECT_INPUT_PATH)
    subject = os.path.basename(SUBJECT_NORM_PATH)

    # Retrieve the images that should be put through the pipeline
    in_anatomical_image = Node(SelectFiles({"out_file": abspath(dataset["anat"].format(subject=subject, subject_path=SUBJECT_NORM_PATH))}), name="in_anatomical_image")
    in_magnitude_image = Node(SelectFiles({"out_file": abspath(dataset["fmap"]["magnitude"].format(subject=subject, subject_path=SUBJECT_NORM_PATH))}), name="in_magnitude_image")
    in_phase_image = Node(SelectFiles({"out_file": abspath(dataset["fmap"]["phasediff"].format(subject=subject, subject_path=SUBJECT_NORM_PATH))}), name="in_phase_image")
    in_functional_image = Node(SelectFiles({"out_file": abspath(dataset["func"].format(subject=subject, subject_path=SUBJECT_NORM_PATH))}), name="in_functional_image")

    # Creating output directory for the designated subject
    output_dir = os.path.join(SUBJECT_NORM_PATH, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # output files
    out_field_map = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "field_map.nii.gz")), clobber=True), name="out_field_map")
    out_meanf_to_anat = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "func2struct.mat")), clobber=True), name="out_func_to_struct")
    out_b0_d = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "b0_d.nii.gz")), clobber=True), name="out_b0_d")
    out_b0_mask = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "b0_mask.nii.gz")), clobber=True), name="out_b0_mask")
    out_b0_u = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "b0_u.nii.gz")), clobber=True), name="out_b0_u")
    out_t1w = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "T1w.nii.gz")), clobber=True), name="out_t1w")
    out_b0_d_10 = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "b0_d_10.nii.gz")), clobber=True), name="out_b0_d_10")

    """ Function Nodes """
    # Skullstripping and magnitude erode
    anat_skullstrip = Node(fsl.BET(frac=0.5, mask=True), name="anat_skullstrip")
    mag_skullstrip = Node(fsl.BET(frac=0.5, mask=True), name="mag_skullstrip")
    mag_erode = Node(fsl.maths.ErodeImage(), name="mag_erode")
    # Mean Image
    mean_func = Node(fsl.maths.MeanImage(dimension="T"), name="mean_func")
    # Median image
    median_tf = Node(Function(function=GetMedianTF, input_names=["in_file"], output_names=["out_value"]), name="median_tf")
    median_tf_minus_five = Node(Function(function=SubtractFive, input_names=["in_value"], output_names=["out_value"]), name="median_tf_minus_five")
    extract_roi_func_10 = Node(fsl.ExtractROI(t_size=10), name="extract_roi_func_10")
    # Registrations and matrices
    mean_func_to_anat_reg = Node(fsl.epi.EpiREG(), name="mean_func_to_anat_reg")
    invert_mean_func_to_anat_mat = Node(fsl.utils.ConvertXFM(invert_xfm=True), name="invert_mean_func_to_anat_mat")
    mag_to_mean_func_reg = Node(fsl.FLIRT(out_matrix_file="mag_to_mean_func_reg.mat"), name="mag_to_mean_func_reg")
    phase_to_mean_func_reg = Node(fsl.FLIRT(apply_xfm=True), name="phase_to_mean_func_reg")
    # Prepare fieldmap
    prepare_fieldmap = Node(fsl.PrepareFieldmap(delta_TE=2.46, scanner="SIEMENS"), name="prepare_fieldmap")
    # Fugue correction
    fugue_correction = Node(fsl.FUGUE(dell_time=dataset["echospacing"], unwarp_direction=dataset["phaseencodingdirection"]), name="fugue_correction")
    # Skullstrip fugue correction
    fugue_skullstrip = Node(fsl.BET(functional=True), name="fugue_skullstrip")
    # Func skullstrip
    func_skullstrip = Node(fsl.BET(functional=True, mask=True), name="func_skullstrip")
    # Transform and intensity normalizing anat
    transform_anat = Node(fsl.FLIRT(apply_xfm=True), name="transform_anat")
    intensity_normalize_anat = Node(Function(function=IntensityNormalization, input_name=["in_file"], output_names=["out_file"]), name="intensity_normalize_anat")

    """ Workflow """
    # Define workflow
    preprocessing_workflow = Workflow(name="preprocessing_workflow")
    # Connect workflow
    preprocessing_workflow.connect([
        (in_anatomical_image, anat_skullstrip, (["out_file", "in_file"])),                      # Anat skullstripping
        (in_magnitude_image, mag_skullstrip, (["out_file", "in_file"])),                        # Mag skullstripping
        (mag_skullstrip, mag_erode, (["out_file", "in_file"])),                                 # Mag erosion
        (in_functional_image, median_tf, (["out_file", "in_file"])),                            # Extract median tf
        (in_functional_image, extract_roi_func_10, (["out_file", "in_file"])),                  # Extract 10 tf functional image
        (median_tf, median_tf_minus_five, (["out_value", "in_value"])),                         # Subtract five from median
        (median_tf_minus_five, extract_roi_func_10, (["out_value", "t_min"])),                  # Extract the functional timeframe
        (extract_roi_func_10, out_b0_d_10, (["roi_file", "in_file"])), 
        (in_functional_image, mean_func, (["out_file", "in_file"])),                            # Func mean image
        (mag_erode, mag_to_mean_func_reg, (["out_file", "in_file"])),                           # Mag to mean func reg
        (mean_func, mag_to_mean_func_reg, (["out_file", "reference"])),
        (in_phase_image, phase_to_mean_func_reg, (["out_file", "in_file"])),                    # Phase to mean func reg
        (mag_to_mean_func_reg, phase_to_mean_func_reg, (["out_matrix_file", "in_matrix_file"])),
        (mean_func, phase_to_mean_func_reg, (["out_file", "reference"])),
        (in_anatomical_image, mean_func_to_anat_reg, (["out_file", "t1_head"])),                # Mean func to anat reg
        (anat_skullstrip, mean_func_to_anat_reg, (["out_file", "t1_brain"])),
        (mean_func, mean_func_to_anat_reg, (["out_file", "epi"])),
        (mean_func_to_anat_reg, out_meanf_to_anat, (["epi2str.mat","in_file"])),                # Export mean func to anat reg matrix
        (mean_func_to_anat_reg, invert_mean_func_to_anat_mat, (["epi2str.mat", "in_file"])),    # Invert the mean func to anat reg matrix
        (phase_to_mean_func_reg, prepare_fieldmap, (["out_file", "in_phase"])),                 # Prepare the fieldmap
        (mag_to_mean_func_reg, prepare_fieldmap, (["out_file", "in_phase"])),
        (prepare_fieldmap, out_field_map, (["out_fieldmap", "in_file"])),                       # Export the fieldmap
        (in_functional_image, fugue_correction, (["out_file", "in_file"])),                     # Creating ground truth correction
        (prepare_fieldmap, fugue_correction, (["out_fieldmap", "in_file"])),
        (fugue_correction, fugue_skullstrip, (["unwarped_file", "in_file"])),                   # Skullstrip unwarped func
        (fugue_skullstrip, out_b0_u, (["out_file", "in_file"])),                                # Export the skullstripped unwarped output
        (func_skullstrip, out_b0_d, (["out_file", "in_file"])),                                 # Export non-corrected func image
        (func_skullstrip, out_b0_mask, (["out_mask", "in_file"])),                              # Export the functional image mask
        (anat_skullstrip, intensity_normalize_anat, (["out_file", "in_file"])),                 # Intensity normalizing anatomical
        (intensity_normalize_anat, transform_anat, (["out_file", "in_file"])),                  # Transform anatomical image
        (fugue_skullstrip, transform_anat, (["out_file", "reference"])),
        (invert_mean_func_to_anat_mat, transform_anat, (["out_file", "in_matrix_file"])),
        (transform_anat, out_t1w, (["out_file", "in_file"])),                                   # Export t1w image
    ])

    # # Setting up the export nodes for the processed files
    # out_field_map = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "field_map.nii.gz")), clobber=True), name="out_field_map")
    # out_b0_d = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "b0_d.nii.gz")), clobber=True), name="out_b0_d")
    # out_b0_mask = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "b0_mask.nii.gz")), clobber=True), name="out_b0_mask")
    # out_b0_u = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "b0_u.nii.gz")), clobber=True), name="out_b0_u")
    # out_t1w = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "T1w.nii.gz")), clobber=True), name="out_t1w")
    # out_mni_warp = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "T1w_to_MNI_warp.nii.gz")), clobber=True), name="out_mni_warp")
    # # Find bold image median time index, perform single extraction, median 10 step extraction
    # median_tf = Node(Function(function=GetMedianTF, input_names=["in_file"], output_names=["out_value"]), name="median_tf")
    # extract_roi_functional = Node(fsl.ExtractROI(t_size=1), name="extract_roi_functional")
    # median_tf_minus_five = Node(Function(function=SubtractFive, input_names=["in_value"], output_names=["out_value"]), name="median_tf_minus_five")

    # extract_roi_functional_10 = Node(fsl.ExtractROI(t_size=10), name="extract_roi_functional_10")



    # 

    # # Skullstripping bold time series, unwarped functional image, transform to t1, t1 intensity normalization

    # skullstrip_functional_10 = Node(fsl.BET(functional=True, mask=True), name="skullstrip_functional_10")
    # skullstrip_unwarped_functional = Node(fsl.BET(functional=True), name="skullstrip_unwarped_functional")
    
    # transform_t1 = Node(fsl.FLIRT(apply_xfm=True), name="transform_t1")

    # intensity_normalize_t1 = Node(Function(function=IntensityNormalization, input_names=["in_file"], output_names=["out_file"]), name="intensity_normalize_t1")

    # # Affine and non-linear registration to MNI atlas space

    # transform_t1_to_mni = Node(fsl.FLIRT(dof=12, reference=f"{FSL_DIR}/data/standard/MNI152_T1_2mm_brain.nii.gz"), name="transform_t1_to_mni")

    # register_t1_to_mni = Node(fsl.FNIRT(config_file="T1_2_MNI152_2mm", ref_file=f"{FSL_DIR}/data/standard/MNI152_T1_2mm_brain.nii.gz", field_file=True), name="register_t1_to_mni")

    # # Defining the workflow
    # # Computing the median and time sequence for the functional images
    # # Perform the fugue correction
    # # Functional to t1w registration preparation
    # # Skullstripping t1w image and registrering the epi


    # # 10 volumne BOLD is skullstripped (distorted) and exported, the bain mask from it is exported
    # workflow.connect(extract_roi_functional_10, "roi_file", skullstrip_functional_10, "in_file")
    # workflow.connect(skullstrip_functional_10, "out_file", out_b0_d, "in_file")
    # workflow.connect(skullstrip_functional_10, "mask_file", out_b0_mask, "in_file")
    
    # # Skullstripping the fugue corrected functional image, and exporting the output
    # workflow.connect(fugue_correction, "unwarped_file", skullstrip_unwarped_functional, "in_file")
    # workflow.connect(skullstrip_unwarped_functional, "out_file", out_b0_u, "in_file")
    
    # # Intensity normalizing the T1 image, transform it to functional space, skullstripped undistorted is used as t1 reference
    # workflow.connect(skullstrip_t1w, "out_file", intensity_normalize_t1, "in_file")
    # workflow.connect(intensity_normalize_t1, "out_file", transform_t1, "in_file")
    # workflow.connect(skullstrip_unwarped_functional, "out_file", transform_t1, "reference")
   
    # # Inverse registration matrix is used to map t1 to functional space, resampled t1 is exported
    # workflow.connect(invert_registration_matrix, "out_file", transform_t1, "in_matrix_file")
    # workflow.connect(transform_t1, "out_file", out_t1w, "in_file")
   
    # # Normalized t1 used for both affine transformation and nonlinear registration
    # workflow.connect(intensity_normalize_t1, "out_file", transform_t1_to_mni, "in_file")
    # workflow.connect(intensity_normalize_t1, "out_file", register_t1_to_mni, "in_file")
   
    # # Affine transformation matrix from FLIRT, nonlinear warp field is exported
    # workflow.connect(transform_t1_to_mni, "out_matrix_file", register_t1_to_mni, "affine_file")
    # workflow.connect(register_t1_to_mni, "field_file", out_mni_warp, "in_file")
   
    # # Run the entire workflow
    # workflow.run()

"""
Taking subject input path and subject output path, copy the result
"""
def _copy_output_to_new_dir(SUBJECT_INPUT_PATH: str, SUBJECT_OUTPUT_PATH: str):
    # Check if the output path exists, otherwise create
    if not os.path.exists(SUBJECT_OUTPUT_PATH):
        os.makedirs(SUBJECT_OUTPUT_PATH)
    # Copy output from the results and send it to the output path
    shutil.copytree(os.path.join(SUBJECT_INPUT_PATH, "results"), SUBJECT_OUTPUT_PATH, dirs_exist_ok=True)

"""
Taking input/output paths and fsl application dir, convert all the datasets
"""
def _convert_all_datasets(SOURCE_DATASET_ROOT_DIR: str, DEST_DATASET_ROOT_DIR: str, FSL_DIR: str, JSON_PROCESSING_CONFIG_PATH: str):
    # Check if the output directory exists, otherwise create it
    if not os.path.exists(DEST_DATASET_ROOT_DIR):
        os.makedirs(DEST_DATASET_ROOT_DIR)

    # Retrieve and load the datasets from the json file
    with open(JSON_PROCESSING_CONFIG_PATH, "r") as ds_file:
        datasets = json.load(ds_file)

    # Go through each of the datasets
    for dataset_name, dataset in tqdm(datasets.items(), desc="Datasets"):
        # Print statement to keep track of dataset
        print(f'Converting dataset "{dataset_name}"...')
        
        # Create the output directory path
        DATASET_OUTPUT_PATH = os.path.join(DEST_DATASET_ROOT_DIR, dataset_name)
        
        # If the dataset directory doesn't exist, make it
        if not os.path.exists(DATASET_OUTPUT_PATH):
            os.makedirs(DATASET_OUTPUT_PATH)

        # Go through each subject in the dataset
        for SUBJECT_INPUT_PATH in tqdm(glob(f'{SOURCE_DATASET_ROOT_DIR}/{dataset_name}/sub-*/'), desc="Subjects"):
            # Retieve the subject name and subject path
            subject = os.path.basename(os.path.normpath(SUBJECT_INPUT_PATH))
            SUBJECT_OUTPUT_PATH = os.path.join(DATASET_OUTPUT_PATH, subject)
            
            # Define a boolean to determine if all the necessary files already exists
            all_files_exist = os.path.isfile(os.path.join(SUBJECT_OUTPUT_PATH, 'T1w.nii.gz')) and \
                os.path.isfile(os.path.join(SUBJECT_OUTPUT_PATH, 'b0_d.nii.gz')) and \
                os.path.isfile(os.path.join(SUBJECT_OUTPUT_PATH, 'b0_u.nii.gz')) and \
                os.path.isfile(os.path.join(SUBJECT_OUTPUT_PATH, 'b0_mask.nii.gz')) and \
                os.path.isfile(os.path.join(SUBJECT_OUTPUT_PATH, 'T1w_to_MNI_warp.nii.gz')) and \
                os.path.isfile(os.path.join(SUBJECT_OUTPUT_PATH, 'field_map.nii.gz')) and \
                os.path.isfile(os.path.join(SUBJECT_OUTPUT_PATH, 'func2struct.mat'))
            
            # If all the files already exists, continue, otherwise convert the subject data
            if all_files_exist:
                continue
            else:
                # Convert the subject and copy its output to the output directory
                print(f'Processing subject {subject}...')
                _convert_subject(SUBJECT_INPUT_PATH, dataset, FSL_DIR)
                _copy_output_to_new_dir(SUBJECT_INPUT_PATH, SUBJECT_OUTPUT_PATH)
                
        # Copy the participants.tsv file to the processed data output path
        shutil.copy2(
            src=os.path.join(f'{SOURCE_DATASET_ROOT_DIR}/{dataset_name}', 'participants.tsv'),
            dst=os.path.join(DATASET_OUTPUT_PATH, "participants.tsv")
        )

        # Create a dataset_meta.json file containing the echospacing information
        dataset_meta = {"echospacing": dataset["echospacing"]}
        with open(os.path.join(DATASET_OUTPUT_PATH, "dataset_meta.json"), "w") as meta_file: meta_file.write(json.dumps(dataset_meta, indent=4))

"""
Main section for running the processing pipeline
Given a set of raw OpenNeuro datasets with JSON metadata, transform them into a representation that
can be used for training and inference in the model
"""
if __name__ == '__main__':
    # Create a parser and add the necessary arguments
    parser = argparse.ArgumentParser(description="Preprocess neuroimages to a suitable model format for training and inference")
    parser.add_argument("--FSL_DIR", type=str, required=True, help="Path to the FSL application root folder on your local machine")
    parser.add_argument("--SOURCE_DATASET_ROOT_DIR", type=str, required=True, help="Path to the source directory root containing the datasets to process")
    parser.add_argument("--DEST_DATASET_ROOT_DIR", type=str, required=True, help="Path to the destination directory root where the processed datasets should be saved")
    parser.add_argument("--JSON_PROCESSING_CONFIG_PATH", type=str, required=True, help="Path to json file with subsampled file paths and echospacing information")
    args = parser.parse_args()

    # Retrieve the arguments
    FSL_DIR = args.FSL_DIR
    SOURCE_DATASET_ROOT_DIR = args.SOURCE_DATASET_ROOT_DIR
    DEST_DATASET_ROOT_DIR = args.DEST_DATASET_ROOT_DIR
    JSON_PROCESSING_CONFIG_PATH = args.JSON_PROCESSING_CONFIG_PATH

    # Printing out the arguments
    print(f"FSL directory: {FSL_DIR}")
    print(f"Source dataset root directory: {SOURCE_DATASET_ROOT_DIR}")
    print(f"Destination dataset root directory: {DEST_DATASET_ROOT_DIR}")
    print(f"JSON processing configuration path: {JSON_PROCESSING_CONFIG_PATH}\n")

    # Run the processing pipeline
    _convert_all_datasets(
        SOURCE_DATASET_ROOT_DIR=SOURCE_DATASET_ROOT_DIR,
        DEST_DATASET_ROOT_DIR=DEST_DATASET_ROOT_DIR,
        FSL_DIR=FSL_DIR,
        JSON_PROCESSING_CONFIG_PATH=JSON_PROCESSING_CONFIG_PATH
    )
