import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from project.data import data_util


def collect_all_subject_paths(dataset_paths):
    subject_paths = []

    for dataset_path in dataset_paths:
        participants_df = pd.read_csv(os.path.join(dataset_path, 'participants.tsv'), sep='\t')
        for participant in participants_df.participant_id:
            subject_path = os.path.join(dataset_path, participant)
            subject_paths.append(subject_path)

    return subject_paths


def load_data_from_path(subject_path):
    # Get paths
    t1_path = os.path.join(subject_path, 'T1w.nii.gz')
    b0_d_path = os.path.join(subject_path, 'b0_d.nii.gz')
    b0_u_path = os.path.join(subject_path, 'b0_u.nii.gz')
    mask_path = os.path.join(subject_path, 'b0_mask.nii.gz')
    fieldmap_path = os.path.join(subject_path, 'field_map.nii.gz')

    # Get meta information
    dataset_path = Path(subject_path).parent.absolute()
    with open(os.path.join(dataset_path, 'dataset_meta.json')) as f:
        dataset_meta = json.load(f)

    # Get image
    img_t1 = data_util.get_nii_img(t1_path)
    img_b0_d = data_util.get_nii_img(b0_d_path)
    img_b0_u = data_util.get_nii_img(b0_u_path)
    img_mask = data_util.get_nii_img(mask_path)
    img_fieldmap = data_util.get_nii_img(fieldmap_path)[:, :, :, 0]

    '''# Pad array since I stupidly used template with dimensions not factorable by 8
    # Assumes input is (77, 91, 77) and pad to (80, 96, 80) with zeros
    img_t1 = np.pad(img_t1, ((2, 1), (3, 2), (2, 1), (0, 0)), 'constant')
    img_b0_d = np.pad(img_b0_d, ((2, 1), (3, 2), (2, 1), (0, 0)), 'constant')
    img_b0_u = np.pad(img_b0_u, ((2, 1), (3, 2), (2, 1), (0, 0)), 'constant')
    img_mask = np.pad(img_mask, ((2, 1), (3, 2), (2, 1), (0, 0)), 'constant')'''

    number_timesteps = img_b0_d.shape[3]

    # Repeat T1 image if we only have one
    if len(img_t1.shape) == 3:
        img_t1 = np.repeat(img_t1[None, :], number_timesteps, axis=0)
        img_t1 = np.transpose(img_t1, axes=(1, 2, 3, 0))

    if len(img_fieldmap.shape) == 3:
        img_fieldmap = np.repeat(img_fieldmap[None, :], number_timesteps, axis=0)
        img_fieldmap = np.transpose(img_fieldmap, axes=(1, 2, 3, 0))

    '''print(f't1 before padding: {img_t1.shape}')
    print(f'img_b0_d before padding: {img_b0_d.shape}')
    print(f'img_b0_u before padding: {img_b0_u.shape}')
    print(f'img_mask before padding: {img_mask.shape}')
    
    # TODO: Pad or scale everything to (64 x 64 x 36 x n)

    print(f't1 after padding: {img_t1.shape}')
    print(f'img_b0_d after padding: {img_b0_d.shape}')
    print(f'img_b0_u after padding: {img_b0_u.shape}')
    print(f'img_mask after padding: {img_mask.shape}')'''

    # Convert to torch img format
    img_t1 = data_util.nii2torch(img_t1)
    img_b0_d = data_util.nii2torch(img_b0_d)
    img_b0_u = data_util.niiu2torch(img_b0_u)
    img_mask = data_util.niimask2torch(img_mask, repetitions=number_timesteps) != 0
    img_fieldmap = data_util.niiu2torch(img_fieldmap)

    # Normalize data
    img_t1 = data_util.normalize_img(img_t1, 150, 0, 1, -1)  # Based on freesurfers T1 normalization
    max_img_b0_d = np.percentile(img_b0_d, 99)  # This usually makes majority of CSF be the upper bound
    min_img_b0_d = 0  # Assumes lower bound is zero (direct from scanner)
    img_b0_d = data_util.normalize_img(img_b0_d, max_img_b0_d, min_img_b0_d, 1, -1)
    img_b0_u = data_util.normalize_img(img_b0_u, max_img_b0_d, min_img_b0_d, 1, -1)  # Use min() and max() from distorted data

    '''# Set "data" and "target"
    img_data = np.concatenate((img_b0_d, img_t1), axis=1)
    img_target = img_b0_u'''

    img_mask = np.array(img_mask, dtype=np.uint8)

    b0u_affine = nib.load(b0_u_path).affine
    b0u_affine = np.repeat(b0u_affine[None, :], number_timesteps, axis=0)

    fieldmap_affine = nib.load(fieldmap_path).affine
    fieldmap_affine = np.repeat(fieldmap_affine[None, :], number_timesteps, axis=0)

    echo_spacing = np.array(dataset_meta['echoSpacing'])
    echo_spacing = np.repeat(echo_spacing, number_timesteps, axis=0)

    return img_t1, img_b0_d, img_b0_u, img_mask, img_fieldmap, b0u_affine, fieldmap_affine, echo_spacing
