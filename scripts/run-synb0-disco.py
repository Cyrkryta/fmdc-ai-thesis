import os
import time
import json
import subprocess
from os import path
from tqdm import tqdm
import shutil

input_base = path.join("/home/mlc/dev/fmdc/downloads/synb0-disco", "INPUTS")
output_base = path.join("/home/mlc/dev/fmdc/downloads/synb0-disco", "OUTPUTS")
computation_times_file = path.join("/home/mlc/dev/fmdc/downloads/synb0-disco", "computation_times.json")

if path.isfile(computation_times_file):
    with open(computation_times_file, "r") as f:
        computation_times = json.load(f)
else:
    computation_times = {}

def store_computation_times():
    """
    Function for storing computation times
    """
    with open(computation_times_file, "w+") as f:
        json.dump(computation_times, f)
    
# Check if INPUTS and OUTPUTS directories exists
if not path.exists(input_base):
    raise Exception(f"Inputs folder doesn't exists")
if not path.exists(output_base):
    os.makedirs(output_base, exist_ok=True)

# Docker config
license_file = path.join("/home/mlc/dev/fmdc/downloads/synb0-disco", "license.txt")
docker_image = "leonyichencai/synb0-disco:v3.1"
docker_flags = "--stripped"
synb0cnf = "/home/mlc/dev/fmdc/downloads/synb0-disco/synb0.cnf"

# Loop through each dataset folder in INPUTS
overall_start = time.time()
for dataset in tqdm(os.listdir(input_base), desc="Datasets"):
    dataset_input_path = path.join(input_base, dataset)
    dataset_output_path = path.join(output_base, dataset)

    # Continue if not a directory
    if not path.isdir(dataset_input_path):
        continue
    # Create OUTPUT dataset folder if needed
    os.makedirs(dataset_output_path, exist_ok=True)

    # Loop through each subject in the dataset folder
    for subject in tqdm(os.listdir(dataset_input_path), desc=f"Subjects in {dataset}"):
        subject_input_dir = path.join(dataset_input_path, subject)
        subject_output_dir = path.join(dataset_output_path, subject)

        if not path.isdir(subject_input_dir):
            continue

        os.makedirs(subject_output_dir, exist_ok=True)

        for timestep in tqdm(os.listdir(subject_input_dir), desc=f"Timesteps in {subject}"):
            timestep_input_dir = path.join(subject_input_dir, timestep)
            print(timestep_input_dir)
            timestep_output_dir = path.join(subject_output_dir, timestep)
            print(timestep_output_dir)

            docker_command = (
                "docker run --rm "
                f"-v {timestep_input_dir}/:/INPUTS/ "
                f"-v {timestep_output_dir}:/OUTPUTS/ "
                f"-v {license_file}:/extra/freesurfer/license.txt "
                f"-v {synb0cnf}:/extra/synb0.cnf "
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
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time for subject {subject}: {elapsed_time:.2f} seconds")

            keep = {"b0_all.nii.gz", "b0_all_topup.nii.gz"}
            for fname in os.listdir(timestep_output_dir):
                if fname not in keep:
                    target = path.join(timestep_output_dir, fname)
                    try:
                        if path.isdir(target):
                            shutil.rmtree(target)
                        else:
                            os.remove(target)
                        print(f"Removed: {target}")
                    except Exception as e:
                        print(f"Error removing {target}: {e}")

            if f'{dataset}-{subject}' not in computation_times:
                computation_times[f'{dataset}-{subject}'] = {}

            # computation_times[f'{dataset}-{subject}'][f't-{timestep:03d}'] = elapsed_time
            computation_times[f'{dataset}-{subject}'][timestep] = elapsed_time
            store_computation_times()

overall_end = time.time()
print(f"Time passed for subject: {overall_end - overall_start}")


print("All subjects processed.")
