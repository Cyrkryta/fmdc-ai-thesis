import os
import json
import shutil
import random
import nibabel as nib
from tqdm import tqdm
import subprocess

def convert_file_to_niigz(input_file, output_file):
    try:
        img = nib.load(input_file)
        nib.save(img, output_file)
    except Exception as e:
        print(f"Error converting {input_file} to {output_file}: {e}")

def get_anat_path(templates, subject_path, subject_id):
    """
    Retrieve the original anatomical image path from the template.
    """
    anat_template = templates["anat"]
    anat_path = anat_template.replace("{subject_path}", subject_path).replace("{subject}", subject_id)
    return anat_path

def move_subjects():
    """
    Function for moving subjects to prepare the Synb0 directory for docker
    """
    # Create paths
    base_dir = "/home/mlc/dev/fmdc/downloads/synb0-disco"
    input_base = os.path.join(base_dir, "INPUTS")
    output_base = os.path.join(base_dir, "OUTPUTS")
    anat_base = "/home/mlc/dev/fmdc/downloads/datasets/test"
    test_paths_file = "/home/mlc/dev/fmdc/downloads/test-processed-NEW/test_paths.json"
    test_template_file = "/home/mlc/dev/fmdc/downloads/datasets/test/test_process_paths.json"

    # Load the json files for test
    with open(test_paths_file, "r") as f:
        SYNB0_TEST_PATHS = json.load(f)
    with open(test_template_file, "r") as f:
        TEMPLATE_PATHS = json.load(f)

    # Retrieve the subject dirs
    subject_dirs = SYNB0_TEST_PATHS["test_paths"]
    # print(len(subject_dirs))

    # Run through the subjects
    for SUBJECT_PATH in tqdm(subject_dirs, desc="Preparing subjects"):
        dataset = os.path.basename(os.path.dirname(SUBJECT_PATH))
        subject = os.path.basename(SUBJECT_PATH)
        print(f"\nProcessing dataset: {dataset}, subject: {subject}")

        if dataset != "ds005165" or subject != "sub-01":
            continue

        # Check if all datasets are available in the template file
        if dataset not in TEMPLATE_PATHS:
            print(f"Dataset {dataset} not found in template file, skibbing {subject}")
            continue

        # Retrieve the template
        templates = TEMPLATE_PATHS[dataset]
        anat_subject_base = os.path.join(anat_base, dataset, subject)
        anat_path = get_anat_path(templates, anat_subject_base, subject)
        print(f"Anatomical image path: {anat_path}")

        # Define the paths for the subject level images
        b0_d_path = os.path.join(SUBJECT_PATH, "b0_d.nii.gz")
        b0_u_path = os.path.join(SUBJECT_PATH, "b0_u.nii.gz")
        mask_path = os.path.join(SUBJECT_PATH, "b0_mask.nii.gz")

        # Create subject-level INPUTS and OUTPUTS folders
        subject_input_folder = os.path.join(input_base, dataset, subject)
        subject_output_folder = os.path.join(output_base, dataset, subject)
        os.makedirs(subject_input_folder, exist_ok=True)
        os.makedirs(subject_output_folder, exist_ok=True)

        anat_temp = os.path.join(subject_input_folder, "T1_temp.nii.gz")
        if anat_path.endswith('.nii.gz'):
            try:
                shutil.copy2(anat_path, anat_temp)
            except Exception as e:
                print(f"Error copying anatomical image for subject {subject}: {e}")
                continue
        elif anat_path.endswith('.nii'):
            convert_file_to_niigz(anat_path, anat_temp)
        else:
            print(f"Unrecognized anatomical image format for {anat_path}. Skipping subject {subject}.")
            continue
        
        # Run FSL's BET to skullstrip the T1 image.
        skullstripped_T1 = os.path.join(subject_input_folder, "T1_brain.nii.gz")
        try:
            bet_cmd = ["bet", anat_temp, skullstripped_T1, "-f", "0.5", "-g", "0"]
            subprocess.run(bet_cmd, check=True)
            print(f"Skullstripped T1 image saved as {skullstripped_T1}")
        except Exception as e:
            print(f"Error during skullstripping with BET for subject {subject}: {e}")
            continue
        # Optionally, remove the temporary file.
        if os.path.exists(anat_temp):
            os.remove(anat_temp)

        # Load the 4D BOLD distorted image
        try:
            b0_d = nib.load(b0_d_path)
        except Exception as e:
            print(f"Error loading BOLD image {b0_d_path}: {e}")
        
        # Check if the image is 4D
        b0_d_data = b0_d.get_fdata()
        if b0_d_data.ndim != 4:
            print(f"Expected a 4D BOLD image but got shape {b0_d_data.shape} for subject {subject}")
            continue

        # Retrieve number of time steps
        number_of_timesteps = b0_d_data.shape[3]
        print(f"Found {number_of_timesteps} timesteps for subject {subject}")

        # Process each of the timesteps
        for t in range(number_of_timesteps):
            # Create the timestep output folder
            timestep_folder = f"t-{t:03d}"
            timestep_input_dir = os.path.join(subject_input_folder, timestep_folder)
            timestep_output_folder = os.path.join(subject_output_folder, timestep_folder)
            os.makedirs(timestep_input_dir, exist_ok=True)
            os.makedirs(timestep_output_folder, exist_ok=True)

            # Slice the image
            b0_d_data_t = b0_d_data[:, :, :, t]
            b0_d_data_t_img = nib.Nifti1Image(b0_d_data_t, b0_d.affine, header=b0_d.header)
            b0_d_data_t_dest_path = os.path.join(timestep_input_dir, "b0.nii.gz")
            nib.save(b0_d_data_t_img, b0_d_data_t_dest_path)
            print(f"Saved timestep {timestep_folder} slice to {b0_d_data_t_dest_path}")

            # Copy the skullstripped T1 image into the timepoint folder.
            timestep_T1_dest = os.path.join(timestep_input_dir, 'T1.nii.gz')
            try:
                shutil.copy2(skullstripped_T1, timestep_T1_dest)
                print(f"Copied skullstripped T1 to {timestep_T1_dest}")
            except Exception as e:
                print(f"Error copying skullstripped T1 for {timestep_folder} of subject {subject}: {e}")

            # Copy the undistorted image.
            timestep_u_dest = os.path.join(timestep_input_dir, 'b0_u.nii.gz')
            if os.path.exists(b0_u_path):
                try:
                    shutil.copy2(b0_u_path, timestep_u_dest)
                    print(f"Copied undistorted image from {b0_u_path} to {timestep_u_dest}")
                except Exception as e:
                    print(f"Error copying undistorted image for {timestep_folder} of subject {subject}: {e}")
            else:
                print(f"Undistorted image not found for {SUBJECT_PATH} during timestep processing")

            # Copy the mask image.
            timestep_mask_dest = os.path.join(timestep_input_dir, 'b0_mask.nii.gz')
            if os.path.exists(mask_path):
                try:
                    shutil.copy2(mask_path, timestep_mask_dest)
                    print(f"Copied mask image from {mask_path} to {timestep_mask_dest}")
                except Exception as e:
                    print(f"Error copying mask image for {timestep_folder} of subject {subject}: {e}")
            else:
                print(f"Mask image not found for {SUBJECT_PATH} during timestep processing")

            # Copy the acquisition parameters file.
            acqparams_src = os.path.join(base_dir, 'acqparams.txt')
            acqparams_dest = os.path.join(timestep_input_dir, 'acqparams.txt')

            if os.path.exists(acqparams_src):
                try:
                    shutil.copy2(acqparams_src, acqparams_dest)
                    print(f"Copied acqparams.txt from {acqparams_src} to {acqparams_dest}")
                except Exception as e:
                    print(f"Error copying acqparams.txt for {timestep_folder} of subject {subject}: {e}")
            else:
                print(f"acqparams.txt not found at {acqparams_src}")
            
            print(f"Prepared data for subject {subject}, timestep {timestep_folder}")

        # Once all timepoints are processed, remove the skullstripped T1 file.
        if os.path.exists(skullstripped_T1):
            os.remove(skullstripped_T1)
            print(f"Removed temporary skullstripped T1 file: {skullstripped_T1}")

if __name__ == "__main__":
    move_subjects()
