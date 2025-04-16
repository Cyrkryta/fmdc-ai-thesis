# Import necessary dependencies
import os
import json
import shutil
import random
import nibabel as nib
from tqdm import tqdm
import subprocess

# Convert function
def convert_file_to_niigz(input_file_path, output_file_path):
    """
    Function for converting file to nii.gz if it isn't already
    nii.gz is expected by the baseline
    """
    try:
        img = nib.load(input_file_path)
        nib.save(img, output_file_path)
    except Exception as e:
        print(f"Error converting input file to 'nii.gz' format")

# Anat retrieval function
def get_anat_path(templates, subject_path, subject_id):
    """
    Function for retrieving the original t1w image path
    Needed for running the baseline properly
    """
    anat_template = templates["anat"]
    anat_path = anat_template.replace("{subject_path}", subject_path).replace("{subject}", subject_id)
    return anat_path

# Move subjects function
def move_subjects():
    """
    Function for moving subjects to the expected test folder (downloads/synb0-disco)
    """
    # Define all paths to be used
    base_dir = "/indirect/student/magnuschristensen/dev/fmdc/downloads/synb0-disco"
    input_base = os.path.join(base_dir, "INPUTS")
    output_base = os.path.join(base_dir, "OUTPUTS")
    anat_base = "/indirect/student/magnuschristensen/dev/fmdc/downloads/original-datasets/test"
    test_paths_file = "/indirect/student/magnuschristensen/dev/fmdc/data-paths/test_paths.json"
    test_temp_file = "/indirect/student/magnuschristensen/dev/fmdc/downloads/original-datasets/test/test_process_paths.json"

    # Load the json files for test
    with open(test_paths_file, "r") as f:
        SYNB0_TEST_PATHS = json.load(f) # All of the test paths

    with open(test_temp_file, "r") as f:
        TEMPLATE_PATHS = json.load(f) # Templates to retrieve original data
    
    # Retrieve all of the subject directories
    subject_dirs = SYNB0_TEST_PATHS["test_paths"]
    print(f"# of subject directories found: {len(subject_dirs)}")

    # Set to only retrieve a single subject (OBS! - Remove if all subjects are desired for processing)
    processed_datasets = set()

    # Go through each of the subjects
    # for SUBJECT_PATH in tqdm(subject_dirs[:1], desc="Preparing subjects"):
    for SUBJECT_PATH in tqdm(subject_dirs, desc="Preparing subjects"):
        dataset = os.path.basename(os.path.dirname(SUBJECT_PATH))
        subject = os.path.basename(SUBJECT_PATH)
        print(f"\nPreparing dataset: {dataset}, Subject: {subject}")

        # Comment if all subjects are desired
        if dataset in processed_datasets:
            continue
        else:
            processed_datasets.add(dataset)

        # Check if the dataset is available in the template file
        if dataset not in TEMPLATE_PATHS:
            print(f"\nDataset {dataset} not found in template file, skipping {subject}")
            continue
        else:
            print(f"\nFound {dataset} in template!!!")

        # Retrieve the dataset template
        templates = TEMPLATE_PATHS[dataset]
        anat_subject_base = os.path.join(anat_base, dataset, subject)
        anat_full_path = get_anat_path(templates, anat_subject_base, subject)
        print(f"\nFull anat image path: {anat_full_path}")

        # Subject specific paths
        b0_d_path = os.path.join(SUBJECT_PATH, "b0_d.nii.gz")
        b0_u_path = os.path.join(SUBJECT_PATH, "b0_u.nii.gz")
        mask_path = os.path.join(SUBJECT_PATH, "b0_mask.nii.gz")
        print(f"\nDistorted path: {b0_d_path}")
        print(f"Undistorted path: {b0_u_path}")
        print(f"Mask path: {mask_path}")

        # Create subject level INPUTS and OUTPUTS folders
        subject_input_folder = os.path.join(input_base, dataset, subject)
        subject_output_folder = os.path.join(output_base, dataset, subject)
        print(f"\nSubject INPUTS folder: {subject_input_folder}")
        print(f"Subject OUTPUTS folder: {subject_output_folder}")
        os.makedirs(subject_input_folder, exist_ok=True)
        os.makedirs(subject_output_folder, exist_ok=True)

        # Retrieve and skullstrip anatomical image
        anat_temp = os.path.join(subject_input_folder, "T1_temp.nii.gz")
        if anat_full_path.endswith(".nii.gz"):
            try:
                shutil.copy2(anat_full_path, anat_temp)
            except Exception as e:
                print(f"Error copying anatomical image for subject {subject}: {e}")
                continue
        elif anat_full_path.endswith(".nii"):
            convert_file_to_niigz(anat_full_path, anat_temp)
        else:
            print(f"Couldn't recognize anatomical image for {anat_full_path}... Skipping {subject}")
            continue

        # Run FSL's BET to skullstrip the T1 image
        skullstripped_T1 = os.path.join(subject_input_folder, "T1_brain.nii.gz")
        try:
            print("Skullstripping...")
            bet_cmd = ["bet", anat_temp, skullstripped_T1, "-f", "0.5", "-g", "0"]
            subprocess.run(bet_cmd, check=True)
            print(f"Skullstripped T1 image saved as {skullstripped_T1}")
        except Exception as e:
            print(f"Error during skullstripping with BET for subject {subject}: {e}")
            continue

        # Remove the temporary anat file
        if os.path.exists(anat_temp):
            print(f"Removing temp file at {anat_temp}")
            try:
                os.remove(anat_temp)
            except Exception as e:
                print(f"Couldn't remove file {anat_temp}")

        # Copy the distorted 4D BOLD
        try:
            b0d = nib.load(b0_d_path)
        except Exception as e:
            print(f"Error loading distorted BOLD image with nib {b0_d_path}: {e}")

        # Check if the image is 4D
        b0d_data = b0d.get_fdata()
        if b0d_data.ndim != 4:
            print(f"Expected a 4D BOLD image but got shape {b0d_data.shape} for subject {subject}\nSkipping...")
            continue

        # Calculate the number of timesteps
        number_of_timesteps = b0d_data.shape[3]
        print(f"Found {number_of_timesteps} timesteps for subject {subject}")

        # Go through the timesteps and save volumes in separate folders
        for t in range(number_of_timesteps):
            # Create the time-step folders
            timestep_folder = f"t-{t:03d}"
            timestep_input_dir = os.path.join(subject_input_folder, timestep_folder)
            timestep_output_dir = os.path.join(subject_output_folder, timestep_folder)
            os.makedirs(timestep_input_dir, exist_ok=True)
            os.makedirs(timestep_output_dir, exist_ok=True)

            # Slice the image at the timestep
            b0d_data_t = b0d_data[:, :, :, t]
            b0d_data_t_img = nib.Nifti1Image(b0d_data_t, b0d.affine, b0d.header)
            b0d_data_t_destpath = os.path.join(timestep_input_dir, "b0.nii.gz")
            nib.save(b0d_data_t_img, b0d_data_t_destpath)
            print(f"Saved timestep {timestep_folder} slice to {b0d_data_t_destpath}")

            # Copying the skullstripped image into the folder
            timestep_T1_dest = os.path.join(timestep_input_dir, "T1.nii.gz")
            try:
                shutil.copy2(skullstripped_T1, timestep_T1_dest)
                print(f"Copied skullstripped T1 to {timestep_T1_dest}")
            except Exception as e:
                print(f"Error copying skullstripped T1 for {timestep_folder} of subject {subject}: {e}")

            # Copy the undistorted image
            timestep_b0u_dest = os.path.join(timestep_input_dir, "b0_u.nii.gz")
            if os.path.exists(b0_u_path):
                try:
                    shutil.copy2(b0_u_path, timestep_b0u_dest)
                    print(f"Copied undistorted image from {b0_u_path} to {timestep_b0u_dest}")
                except Exception as e:
                    print(f"Error copying undistorted image for {timestep_folder} of subject {subject}: {e}")

            # Copy the mask
            timestep_mask_dest = os.path.join(timestep_input_dir, "b0_mask.nii.gz")
            if os.path.exists(mask_path):
                try:
                    shutil.copy2(mask_path, timestep_mask_dest)
                    print(f"Copied mask from {mask_path} to {timestep_mask_dest}")
                except Exception as e:
                    print(f"Error copying undistorted image for {timestep_folder} of subject {subject}: {e}")

            # Copy the acquisition parameters file
            # acqparams_src = os.path.join(base_dir, "acqparams.txt")
            # acqparams_dest = os.path.join(timestep_input_dir, "acqparams.txt")
            dataset_acqparams_src = os.path.join(base_dir, f"{dataset}acqparams.txt")
            acqparams_dest = acqparams_dest = os.path.join(timestep_input_dir, "acqparams.txt")

            if os.path.exists(dataset_acqparams_src):
                try:
                    shutil.copy2(dataset_acqparams_src, acqparams_dest)
                    print(f"Copied acqparams.txt from {dataset_acqparams_src} to {acqparams_dest}")
                except Exception as e:
                    print(f"Error copying acqparams.txt for {timestep_folder} of subject {subject}: {e}")
            else:
                print(f"acqparams.txt not found at {dataset_acqparams_src}")
            
            # Print finished message
            print(f"DONE: {subject} - {timestep_folder}")

        # Remove the skullstripped T1 file
        if os.path.exists(skullstripped_T1):
            os.remove(skullstripped_T1)
            print(f"Removed temporary skullstripped T1 file")

# Main runnable
if __name__ == "__main__":
    move_subjects()
