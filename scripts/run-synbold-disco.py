import os
import time
import json
import subprocess
from os import path
from tqdm import tqdm
from nipype.interfaces.fsl import BET

# Base directories (assumed to be in the current working directory)
cwd = os.getcwd()
comp_times_base = "/home/mlc/dev/fmdc/downloads/synbold-disco/PA"
root_dir = "/home/mlc/dev/fmdc/downloads/synbold-disco/PA"
input_base = path.join(root_dir, 'INPUTS')
output_base = path.join(root_dir, 'OUTPUTS')
computation_times_file = path.join(comp_times_base, 'computation_times.json')

# Load existing computation times or initialize dictionary
if path.isfile(computation_times_file):
    with open(computation_times_file, 'r') as f:
        computation_times = json.load(f)
else:
    computation_times = {}

def store_computation_times():
    with open(computation_times_file, 'w+') as f:
        json.dump(computation_times, f)

# Check that INPUTS and OUTPUTS directories exist
if not path.exists(input_base):
    raise Exception(f"INPUTS folder not found at {input_base}")
if not path.exists(output_base):
    os.makedirs(output_base, exist_ok=True)

# Docker configuration
license_file = path.join(cwd, 'baseline-files', 'license.txt')  # Adjust as needed
docker_image = "ytzero/synbold-disco:v1.4"
docker_flags = "--motion_corrected --skull_stripped"

# Loop through each dataset folder in INPUTS
for dataset in tqdm(os.listdir(input_base)):
    # Create input and output paths
    dataset_input_path = path.join(input_base, dataset)
    dataset_output_path = path.join(output_base, dataset)

    # Continue if it is not a directory
    if not path.isdir(dataset_input_path):
        continue

    # Create OUTPUT if needed
    os.makedirs(dataset_output_path, exist_ok=True)

    # Loop through each subject in the folder
    for subject in tqdm(os.listdir(dataset_input_path)):
        # Create input and outputs subject paths
        subject_input_dir = path.join(dataset_input_path, subject)
        subject_output_dir = path.join(dataset_output_path, subject)

        # Continue if it is not a directory
        if not path.isdir(subject_input_dir):
            continue

        os.makedirs(subject_output_dir, exist_ok=True)

        t1_path = path.join(subject_input_dir, 'T1.nii.gz')
        if os.path.exists(t1_path):
            print("Skullstripping T1 image using BET...")
            # Set output file name for skullstripped image.
            skullstripped_t1 = t1_path.replace('.nii.gz', '_brain.nii.gz')
            bet = BET(in_file=t1_path, out_file=skullstripped_t1, frac=0.5, vertical_gradient=0.0, mask=True)
            try:
                bet_result = bet.run()
                # Replace the original T1 image with the skullstripped version.
                if path.exists(skullstripped_t1):
                    os.rename(skullstripped_t1, t1_path)
                    print("Skullstripping complete.")
                else:
                    print("Skullstripping did not produce an output; proceeding with original T1.")
            except Exception as e:
                print(f"Error during skullstripping T1: {e}")
        else:
            print(f"T1 image not found at {t1_path}, skipping skullstripping.")

        print(f"Processing subject {subject} from dataset {dataset}")

        # Build the docker command
        docker_command = (
            "docker run --rm "
            f"-v {subject_input_dir}:/INPUTS/ "  # Mount the subject's INPUT folder.
            f"-v {subject_output_dir}:/OUTPUTS/ " # Mount the corresponding OUTPUT folder.
            f"-v {license_file}:/opt/freesurfer/license.txt "  # Mount the license file.
            "-v /tmp:/tmp "  # Mount /tmp if needed.
            "--user $(id -u):$(id -g) "
            f"{docker_image} "
            f"{docker_flags}"
        )

        print("Running docker command:")
        print(docker_command)

        start_time = time.time()
        process = subprocess.Popen(docker_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            print(line.decode('utf8').strip())
        process.wait()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time for subject {subject}: {elapsed_time:.2f} seconds")

        # Record the computation time.
        key = f"{dataset}-{subject}"
        computation_times[key] = elapsed_time
        store_computation_times()

print("All subjects processed.")