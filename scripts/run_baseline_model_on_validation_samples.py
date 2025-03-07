# Importing all of the ncessary dependencies
import json
import os
from os import path
import shutil
import nibabel as nib
import numpy as np
from tqdm import tqdm
import subprocess
import time

# Defining the subject paths that should be used for inference
subject_paths_json = 'val_paths_single.json'

# Naming input and output paths to docker expected format
input_dir = path.join(os.getcwd(), 'INPUTS')
output_dir = path.join(os.getcwd(), 'OUTPUTS')

# Creating file for storing computation times
computation_times_file = path.join(os.getcwd(), 'computation_times.json')
if path.isfile(computation_times_file):
    with open(computation_times_file, 'r') as f:
        computation_times = json.load(f)
else:
    computation_times = {}

# Function for storing the computation times
def _store_computation_times():
    with open(computation_times_file, 'w+') as f:
        json.dump(computation_times, f)

# Checking if either the INPUT or OUTPUT folders exist
# Otherwise, make the directories
if path.exists(input_dir) or path.exists(output_dir):
    raise Exception('Either the INPUT or OUTPUT directory already exists, exiting')
else:
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

# Open the .json file with the subjects
with open(subject_paths_json, 'r') as f:
    subject_paths = json.load(f)['val_paths']

# Go through each subjec path and apply the baseline
for subject_path in tqdm(subject_paths):
    # Split the paths into dataset and subject
    dataset = subject_path.split('/')[-2]
    subject = subject_path.split('/')[-1]
    print(f'Processing subject {subject} from dataset {dataset}')

    # Define the input and output subject directories
    subject_input_dir = path.join(input_dir, dataset, subject)
    subject_output_dir = path.join(output_dir, dataset, subject)

    # Load the subjects' distorted image and calculate the timesteps
    b0_img = nib.load(path.join(subject_path, 'b0_d.nii.gz'))
    unsliced_b0 = np.array(b0_img.dataobj)
    number_of_timesteps = unsliced_b0.shape[3]
    print(f"Creating {number_of_timesteps} timesteps...")

    # Setup each of the timesteps
    # for timestep in range(number_of_timesteps):
    for timestep in range(number_of_timesteps)[:1]:
        # Print what timestep is being undistorted
        print(f"Undistorting {subject} at timestep {timestep}/{number_of_timesteps}")

        # Define and make the timestep directories
        timestep_input_dir = path.join(subject_input_dir, f't-{timestep:03d}')
        timestep_output_dir = path.join(subject_output_dir, f't-{timestep:03d}')
        os.makedirs(timestep_input_dir, exist_ok=True)
        os.makedirs(timestep_output_dir, exist_ok=True)

        # Retrieve the slice for the corresponding time step
        b0_slice = unsliced_b0[:, :, :, timestep]
        b0_slice_img = nib.Nifti1Image(b0_slice, b0_img.affine, header=b0_img.header)
        
        # Save the distorted slice in the corresponding timestep folder as 'b0.nii.gz'
        nib.save(b0_slice_img, path.join(timestep_input_dir, 'b0.nii.gz'))

        # Copy the files needed for the basline to the corresponding folder
        shutil.copyfile(path.join(subject_path, 'T1w.nii.gz'), path.join(timestep_input_dir, 'T1.nii.gz'))
        shutil.copyfile(path.join(subject_path, 'b0_mask.nii.gz'), path.join(timestep_input_dir, 'b0_mask.nii.gz'))
        shutil.copyfile(path.join(subject_path, 'b0_u.nii.gz'), path.join(timestep_input_dir, 'b0_u.nii.gz'))
        shutil.copyfile('acqparams.txt', path.join(timestep_input_dir, 'acqparams.txt'))
        



# allow_existing = False

# input_dir = path.join(os.getcwd(), 'INPUTS')
# output_dir = path.join(os.getcwd(), 'OUTPUTS')

# computation_times_file = path.join(os.getcwd(), 'computation_times.json')
# if path.isfile(computation_times_file):
#     with open(computation_times_file, 'r') as f:
#         computation_times = json.load(f)
# else:
#     computation_times = {}

# def _store_computation_times():
#     with open(computation_times_file, 'w+') as f:
#         json.dump(computation_times, f)

# if not allow_existing and (path.exists(input_dir) or path.exists(output_dir)):
#     raise Exception('Either input or output directory already exists, exiting!')

# os.makedirs(input_dir, exist_ok=True)
# os.makedirs(output_dir, exist_ok=True)

# with open('val_paths_single.json', 'r') as f:
#     val_paths = json.load(f)['val_paths']

# for p in tqdm(val_paths):
#     dataset = p.split('/')[-2]
#     subject = p.split('/')[-1]

#     print(f'Processing subject {subject} from dataset {dataset}')

#     subject_input_dir = path.join(input_dir, dataset, subject)
#     subject_output_dir = path.join(output_dir, dataset, subject)

#     fmri_img = nib.load(path.join(p, 'b0_d.nii.gz'))
#     unsliced_fmri = np.array(fmri_img.dataobj)
#     unsliced_fmri = np.array(fmri_img.dataobj)

#     number_of_timesteps = unsliced_fmri.shape[3]
#     number_of_timesteps = unsliced_fmri.shape[3] 

#     for t in range(number_of_timesteps):
#     for t in range(number_of_timesteps)[:2]: # Only two timesteps
#     timesteps_holder = 2
#     for t in range(timesteps_holder):
#         # print(f'Undistorting timestep {t} of {2}...')
#         print(f'Undistorting timestep {t} of {timesteps_holder}')

#         # timestep_input_dir = path.join(subject_input_dir, f't-{t:03d}')
#         subject_input_dir = '/home/mlc/dev/fmdc/fmdc-ai-thesis/scripts/INPUTS/ds001454/sub-18'
#         timestep_input_dir = path.join(subject_input_dir, f't-{t:03d}')
#         timestep_output_dir = path.join(subject_input_dir, f't-{t:03d}')
#         # timestep_output_dir = path.join(subject_output_dir, f't-{t:03d}')

#         # os.makedirs(timestep_input_dir, exist_ok=True)
#         # os.makedirs(timestep_output_dir, exist_ok=True)

#         # fmri_slice = unsliced_fmri[:, :, :, t]
#         # fmri_slice_img = nib.Nifti1Image(fmri_slice, fmri_img.affine, header=fmri_img.header)
#         # nib.save(fmri_slice_img, path.join(timestep_input_dir, 'b0.nii.gz'))

#         # shutil.copyfile(path.join(p, 'T1w.nii.gz'), path.join(timestep_input_dir, 'T1.nii.gz'))
#         # shutil.copyfile(path.join(p, 'b0_mask.nii.gz'), path.join(timestep_input_dir, 'b0_mask.nii.gz'))
#         # shutil.copyfile(path.join(p, 'b0_u.nii.gz'), path.join(timestep_input_dir, 'b0_u.nii.gz'))
#         # shutil.copyfile('acqparams.txt', path.join(timestep_input_dir, 'acqparams.txt'))

#         # if path.exists(path.join(timestep_output_dir, 'b0_all_topup.nii.gz')):
#         #     print('Skipping because it already exists!')
#         #     continue
        
#         # Start the time for analysis
#         start = time.time()

#         # Path to license and config files
#         license_file = '/home/mlc/dev/fmdc/fmdc-ai-thesis/scripts/baseline-files/license.txt'
#         synb0_cnf_file = '/home/mlc/dev/fmdc/fmdc-ai-thesis/scripts/baseline-files/synb0.cnf'

#         # Define the command
#         command = (
#         "docker run --rm \\\n"
#         f"  -v {timestep_input_dir}:/INPUTS/ \\\n"                  # Input data
#         f"  -v {timestep_output_dir}:/OUTPUTS/ \\\n"                # Output data
#         f"  -v {license_file}:/extra/freesurfer/license.txt \\\n"   # Freesurfer license
#         f"  -v {synb0_cnf_file}:/extra/synb0.cnf \\\n"              # Topup configuration file
#         "  --user $(id -u):$(id -g) \\\n"                           # User information
#         "  leonyichencai/synb0-disco:v3.1 \\\n"                     # Docker image
#         "  --stripped"                                              # Images are already skullstripped
#         )

#         # Run the process 
#         with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
#             for line in process.stdout:
#                 print(line.decode('utf8'))

#         # Stop the time
#         end = time.time()
        
#         # Dataset and subject
#         dataset = 'ds001454'
#         subject = 'sub-18'
#         # Add the subject to computation times
#         if f'{dataset}-{subject}' not in computation_times:
#             computation_times[f'{dataset}-{subject}'] = {}

#         # Calculate and store the computation times
#         computation_times[f'{dataset}-{subject}'][f't-{t:03d}'] = end - start
#         _store_computation_times()

#         # # Add the subject to computation times
#         # if f'{dataset}-{subject}' not in computation_times:
#         #     computation_times[f'{dataset}-{subject}'] = {}

#         # # Calculate and store the computation times
#         # computation_times[f'{dataset}-{subject}'][f't-{t:03d}'] = end - start
#         # _store_computation_times()

# # UNSLICED_B0_PATH = f'/home/mlc/dev/fmdc/fmdc-ai-thesis/scripts/INPUTS/b0_unsliced.nii.gz'
# # unsliced_b0 = nib.load(UNSLICED_B0_PATH)
# # unsliced_b0_array = np.array(unsliced_b0.dataobj)
# # b0_slice = unsliced_b0_array[:, :, :, unsliced_b0_array.shape[3] // 2]
# # b0_slice_img = nib.Nifti1Image(b0_slice, unsliced_b0.affine, header=unsliced_b0.header)
# # nib.save(b0_slice_img, os.path.join('/home/mlc/dev/fmdc/fmdc-ai-thesis/scripts/INPUTS', 'b0.nii.gz'))

# # license_file = '/home/mlc/dev/fmdc/fmdc-ai-thesis/scripts/baseline-files/license.txt'
# # synb0_cnf_file = '/home/mlc/dev/fmdc/fmdc-ai-thesis/scripts/baseline-files/synb0.cnf'

# # # Define the command
# # command = (
# #     "docker run --rm \\\n"
# #     # f"  -e CUDA_VISIBLE_DEVICES='' \\\n"
# #     f"  -v /home/mlc/dev/fmdc/fmdc-ai-thesis/scripts/INPUTS/:/INPUTS/ \\\n"                  # Input data
# #     f"  -v /home/mlc/dev/fmdc/fmdc-ai-thesis/scripts/OUTPUTS/:/OUTPUTS/ \\\n"                # Output data
# #     f"  -v {license_file}:/extra/freesurfer/license.txt \\\n"   # Freesurfer license
# #     f"  -v {synb0_cnf_file}:/extra/synb0.cnf \\\n"              # Topup configuration file
# #     "  --user $(id -u):$(id -g) \\\n"                           # User information
# #     "  leonyichencai/synb0-disco:v3.1 \\\n"                     # Docker image
# #     "  --stripped"                                              # Images are already skullstrippeds
# # )

# # # Run the process 
# # with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
# #     for line in process.stdout:
# #         print(line.decode('utf8'))
