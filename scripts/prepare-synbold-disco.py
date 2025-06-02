import os
import json
import shutil
import random
import nibabel as nib
from tqdm import tqdm

def convert_file_to_niigz(input_file, output_file):
    try:
        img = nib.load(input_file)
        nib.save(img, output_file)
    except Exception as e:
        print(f"Error converting {input_file} to {output_file}: {e}")

def get_anat_path(templates, subject_path, subject_id):
    anat_template = templates["anat"]
    anat_path = anat_template.replace("{subject_path}", subject_path).replace("{subject}", subject_id)
    return anat_path


def move_subjects():
    base_dir = "/home/mlc/dev/fmdc/downloads/synbold-disco/PA"
    input_base = os.path.join(base_dir, "INPUTS")
    output_base = os.path.join(base_dir, "OUTPUTS")
    # anat_base = "/home/mlc/dev/fmdc/downloads/datasets/test"
    anat_base = "/home/mlc/dev/fmdc/downloads/datasets/trainval"

    # Load the JSON files
    # json_test_paths = "/home/mlc/dev/fmdc/downloads/test-processed-NEW/test_paths.json"
    json_test_paths = "/home/mlc/dev/fmdc/downloads/test-processed-NEW/test-processed-PA/pa_test_paths.json"

    # json_temp_path = "/home/mlc/dev/fmdc/downloads/datasets/test/test_process_paths.json"
    json_temp_path = "/home/mlc/dev/fmdc/downloads/datasets/trainval/trainval_process_paths.json"
    
    with open(json_test_paths, "r") as f:
        SYNBOLD_TEST_PATHS = json.load(f)
    with open(json_temp_path, "r") as f:
        TEST_PROCESS_PATHS = json.load(f)

    subject_dirs = SYNBOLD_TEST_PATHS["test_paths"]

    # Go through all of the subjects
    for SUBJECT_PATH in tqdm(subject_dirs, desc="Processing subjects"):
        dataset = os.path.basename(os.path.dirname(SUBJECT_PATH))
        subject = os.path.basename(SUBJECT_PATH)

        print(dataset, subject)

        if dataset not in TEST_PROCESS_PATHS:
            print(f"Dataset {dataset} not found in test-process-paths.json, skipping {SUBJECT_PATH}")
            continue

        templates = TEST_PROCESS_PATHS[dataset]
        # anat_template = templates["anat"]
        # anat_path = anat_template.replace("{subject_path}", SUBJECT_PATH).replace("{subject}", subject)
        anat_subject_base = os.path.join(anat_base, dataset, subject)
        anat_path = get_anat_path(templates, anat_subject_base, subject)
        # anat_path = get_anat_path(templates, SUBJECT_PATH, subject)
        print(anat_path)
        # print(templates)

        b0_d_path = os.path.join(SUBJECT_PATH, "b0_d.nii.gz")
        b0_u_path = os.path.join(SUBJECT_PATH, "b0_u.nii.gz")
        mask_path = os.path.join(SUBJECT_PATH, "b0_mask.nii.gz")

        input_subject_folder = os.path.join(input_base, dataset, subject)
        output_subject_folder = os.path.join(output_base, dataset, subject)
        os.makedirs(input_subject_folder, exist_ok=True)
        os.makedirs(output_subject_folder, exist_ok=True)

        # Define destination file names within the INPUT folder.
        anat_dest = os.path.join(input_subject_folder, "T1.nii.gz")
        bold_d_dest = os.path.join(input_subject_folder, "BOLD_d.nii.gz")
        bold_u_dest = os.path.join(input_subject_folder, "b0_u.nii.gz")
        mask_dest = os.path.join(input_subject_folder, "b0_mask.nii.gz")

        # Process the anatomical image (structural).
        if anat_path.endswith('.nii.gz'):
            try:
                shutil.copy2(anat_path, anat_dest)
                print(f"Copied anatomical image from {anat_path} to {anat_dest}")
            except Exception as e:
                print(f"Error copying anatomical image for {SUBJECT_PATH}: {e}")
        elif anat_path.endswith('.nii'):
            convert_file_to_niigz(anat_path, anat_dest)
        else:
            print(f"Unrecognized anatomical image format for {anat_path}. Skipping conversion.")

        # Process the undistorted ground truth image.
        if os.path.exists(b0_u_path):
            try:
                shutil.copy2(b0_u_path, bold_u_dest)
                print(f"Copied undistorted image from {b0_u_path} to {bold_u_dest}")
            except Exception as e:
                print(f"Error copying undistorted image for {SUBJECT_PATH}: {e}")
        else:
            print(f"Undistorted image {b0_u_path} not found for {SUBJECT_PATH}")

        # Process the distorted functional image.
        if os.path.exists(b0_d_path):
            try:
                shutil.copy2(b0_d_path, bold_d_dest)
                print(f"Copied distorted image from {b0_d_path} to {bold_d_dest}")
            except Exception as e:
                print(f"Error copying distorted image for {SUBJECT_PATH}: {e}")
        else:
            print(f"Distorted image {b0_d_path} not found for {SUBJECT_PATH}")

        # Process the mask image.
        if os.path.exists(mask_path):
            try:
                shutil.copy2(mask_path, mask_dest)
                print(f"Copied mask image from {mask_path} to {mask_dest}")
            except Exception as e:
                print(f"Error copying mask image for {SUBJECT_PATH}: {e}")
        else:
            print(f"Mask image {mask_path} not found for {SUBJECT_PATH}")

if __name__ == "__main__":
    move_subjects()