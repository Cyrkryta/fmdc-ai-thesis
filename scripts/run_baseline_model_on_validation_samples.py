import json
import os
from os import path
import shutil
import nibabel as nib
import numpy as np
from tqdm import tqdm
import subprocess
import time

input_dir = path.join(os.getcwd(), 'INPUTS')
output_dir = path.join(os.getcwd(), 'OUTPUTS')

computation_times_file = path.join(os.getcwd(), 'computation_times.json')
if path.isfile(computation_times_file):
    with open(computation_times_file, 'r') as f:
        computation_times = json.load(f)
else:
    computation_times = {}

def _store_computation_times():
    with open(computation_times_file, 'w+') as f:
        json.dump(computation_times, f)

if path.exists(input_dir) or path.exists(output_dir):
    raise Exception('Either input or output directory already exists, exiting!')

os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

with open('val_paths.json', 'r') as f:
    val_paths = json.load(f)['val_paths']

for p in tqdm(val_paths):
    dataset = p.split('/')[-2]
    subject = p.split('/')[-1]

    print(f'Processing subject {subject} from dataset {dataset}')

    subject_input_dir = path.join(input_dir, dataset, subject)
    subject_output_dir = path.join(output_dir, dataset, subject)

    fmri_img = nib.load(path.join(p, 'b0_d.nii.gz'))
    unsliced_fmri = np.array(fmri_img.dataobj)

    number_of_timesteps = unsliced_fmri.shape[3]

    for t in range(number_of_timesteps):
        print(f'Undistorting timestep {t} of {number_of_timesteps}...')

        timestep_input_dir = path.join(subject_input_dir, f't-{t:03d}')
        timestep_output_dir = path.join(subject_output_dir, f't-{t:03d}')

        os.makedirs(timestep_input_dir, exist_ok=True)
        os.makedirs(timestep_output_dir, exist_ok=True)

        fmri_slice = unsliced_fmri[:, :, :, t]
        fmri_slice_img = nib.Nifti1Image(fmri_slice, fmri_img.affine, header=fmri_img.header)
        nib.save(fmri_slice_img, path.join(timestep_input_dir, 'b0.nii.gz'))

        shutil.copyfile(path.join(p, 'T1w.nii.gz'), path.join(timestep_input_dir, 'T1.nii.gz'))
        shutil.copyfile(path.join(p, 'b0_mask.nii.gz'), path.join(timestep_input_dir, 'b0_mask.nii.gz'))
        shutil.copyfile(path.join(p, 'b0_u.nii.gz'), path.join(timestep_input_dir, 'b0_u.nii.gz'))
        shutil.copyfile('acqparams.txt', path.join(timestep_input_dir, 'acqparams.txt'))

        if path.exists(path.join(timestep_output_dir, 'b0_all_topup.nii.gz')):
            print('Skipping because it already exists!')
            continue

        start = time.time()

        with subprocess.Popen(f'docker run --rm -v {timestep_input_dir}:/INPUTS/  -v {timestep_output_dir}:/OUTPUTS/ -v {path.join(os.getcwd(), "license.txt")}:/extra/freesurfer/license.txt -v {path.join(os.getcwd(), "synb0.cnf")}:/extra/synb0.cnf --user $(id -u):$(id -g) field-map-ai:0.0.1 --stripped', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
            # process.communicate(password.encode())
            for line in process.stdout:
                print(line.decode('utf8'))

        end = time.time()

        if f'{dataset}-{subject}' not in computation_times:
            computation_times[f'{dataset}-{subject}'] = {}

        computation_times[f'{dataset}-{subject}'][f't-{t:03d}'] = end - start
        _store_computation_times()
