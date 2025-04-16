# Dependencies
import os
import time
import json
import subprocess
from os import path
from tqdm import tqdm
import shutil

# Files
input_base = os.path.join("/indirect/student/magnuschristensen/dev/fmdc/downloads/synb0-disco/INPUTS")
output_base = os.path.join("/indirect/student/magnuschristensen/dev/fmdc/downloads/synb0-disco/OUTPUTS")
computation_times_file = os.path.join("/indirect/student/magnuschristensen/dev/fmdc/downloads/synb0-disco", "computation_times.json")

# Check file status
if path.isfile(computation_times_file):
    with open(computation_times_file, "r") as f:
        computation_times = json.load(f)
else:
    computation_times = {}

# Store computation times
def store_computation_times():
    """
    Function for storing computation times
    """
    with open(computation_times_file, "w+") as f:
        json.dump(computation_times, f)

# Check if the INPUTS and OUTPUTS directories exists
if not path.exists(input_base):
    raise Exception(f"Inputs folder doesn't exist!")
if not path.exists(output_base):
    os.makedirs(output_base, exist_ok=True)

# Docker config
license_file = path.join("/indirect/student/magnuschristensen/dev/fmdc/downloads/synb0-disco", "license.txt")
# docker_image = "leonyichencai/synb0-disco:v3.1"
flags = "--stripped"
synb0cnf = "/indirect/student/magnuschristensen/dev/fmdc/downloads/synb0-disco/synb0.cnf"
sif_location = "/indirect/student/magnuschristensen/dev/fmdc/fmdc-ai-thesis/scripts/baseline-files/synb0-disco_v3.1.sif"

# Go through each dataset
for dataset in tqdm(os.listdir(input_base), desc="Datasets"):
    dataset_input_path = os.path.join(input_base, dataset)
    dataset_output_path = os.path.join(output_base, dataset)
    if not path.isdir(dataset_input_path):
        print(f"Couldn't find the path to the dataset... continueing...")
        continue
    os.makedirs(dataset_output_path, exist_ok=True)

    # Go through each subject
    for subject in tqdm(os.listdir(dataset_input_path), desc=f"Subjects in {dataset}"):
        subject_input_dir = os.path.join(dataset_input_path, subject)
        subject_output_dir = os.path.join(dataset_output_path, subject)
        if not path.isdir(subject_input_dir):
            continue
        os.makedirs(subject_output_dir, exist_ok=True)

        # Go through each subject timestep
        for timestep in tqdm(os.listdir(subject_input_dir), desc=f"Timesteps in {subject}"):
            timestep_input_dir = os.path.join(subject_input_dir, timestep)
            timestep_output_dir = os.path.join(subject_output_dir, timestep)

            apptainer_command = (
                "apptainer run -e "
                f"-B {timestep_input_dir}/:/INPUTS/ "
                f"-B {timestep_output_dir}/:/OUTPUTS/ "
                f"-B {license_file}:/extra/freesurfer/license.txt "
                f"-B {synb0cnf}:/extra/synb0.cnf "
                f"{sif_location} {flags}"
            )

            print(f"Running apptainer command:")
            print(apptainer_command)
            start_time = time.time()
            process = subprocess.Popen(apptainer_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in process.stdout:
                print(line.decode("utf8").strip())
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time for subject {subject}: {elapsed_time:.2f} seconds")
            if f"{dataset}-{subject}" not in computation_times:
                computation_times[f"{dataset}-{subject}"] = {}

            computation_times[f"{dataset}-{subject}"][timestep] = elapsed_time
            store_computation_times()

            print(f"Deleting unnecessary files in {timestep_output_dir}")
            # List of files to keep
            keep_files = {"b0_all.nii.gz", "b0_all_topup.nii.gz"}

            # Cleanup after processing each time-point
            for file in os.listdir(timestep_output_dir):
                if file not in keep_files:
                    file_path = os.path.join(timestep_output_dir, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")

            print(f"Cleanup complete in output dir for time-point {timestep}. Only required files remain.")

            # Delete corresponding time-point in INPUTS
            if os.path.exists(timestep_input_dir):
                try:
                    shutil.rmtree(timestep_input_dir)  # Deletes the entire time-point folder
                    print(f"Deleted time-point {timestep_input_dir} from INPUTS")
                except Exception as e:
                    print(f"Error deleting {timestep_input_dir}: {e}")

print("All subjects processed...")

