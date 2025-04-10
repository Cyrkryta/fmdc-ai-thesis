# Importing all of the dependencies
import json
import os
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from project.data import data_util
import torch

"""
Function for collecting all the subjects in the datasets
"""
def collect_all_subject_paths(dataset_paths):
    subject_paths = []

    for dataset_path in dataset_paths:
        participants_df = pd.read_csv(os.path.join(dataset_path, 'participants.tsv'), sep='\t')
        for participant in participants_df.participant_id:
            subject_path = os.path.join(dataset_path, participant)
            subject_paths.append(subject_path)

    return subject_paths

def load_data_from_path_for_test(subject_path):
    """
    Function only loading the de
    """
    # print("BEFORE creating paths")
    # Define the test paths
    b0d_mean_path = os.path.join(subject_path, "b0_d_mean.nii.gz")
    b0d_path = os.path.join(subject_path, "b0_d.nii.gz")
    b0u_path = os.path.join(subject_path, "b0_u.nii.gz")
    b0_mask_path = os.path.join(subject_path, "b0_mask.nii.gz")
    t1w_path = os.path.join(subject_path, "T1w.nii.gz")
    fieldmap_path = os.path.join(subject_path, "field_map.nii.gz")
    # print("AFTER creating paths")

    # Retrieve metadata
    # print("BEFORE dataset meta")
    dataset_path = Path(subject_path).parent.absolute()
    with open(os.path.join(dataset_path, 'dataset_meta.json')) as f:
        dataset_meta = json.load(f)
    # print("AFTER dataset meta")

    # Load the images
    # print("BEFORE loading images")
    t1w = data_util.get_nii_img(t1w_path)
    b0_mask = data_util.get_nii_img(b0_mask_path)
    b0u = data_util.get_nii_img(b0u_path)
    b0d = data_util.get_nii_img(b0d_path)
    fieldmap = data_util.get_nii_img(fieldmap_path)[:, :, :, 0]
    echospacing = dataset_meta["echospacing"]
    phaseencodingdirection = dataset_meta["phaseencodingdirection"]
    # print("AFTER loading images")
    # print("Shapes:")
    # print(f"t1w: {t1w.shape}")
    # print(f"b0 mask: {b0_mask.shape}")
    # print(f"b0u: {b0u.shape}")
    # print(f"b0d: {b0d.shape}")
    # print(f"fieldmap: {fieldmap.shape}")

    # Define the number of timesteps
    number_timesteps = b0d.shape[3]

    # print(f"BEFORE repeating")
    # Repeat to fit timesteps
    if len(t1w.shape) == 3:
        t1w = np.repeat(t1w[None, :], number_timesteps, axis=0)
        t1w = np.transpose(t1w, axes=(1, 2, 3, 0))
    if len(fieldmap.shape) == 3:
        fieldmap = np.repeat(fieldmap[None, :], number_timesteps, axis=0)
        fieldmap = np.transpose(fieldmap, axes=(1, 2, 3, 0))
    # print("AFTER repeating")
    # print("New shapes")
    # print(f"t1w: {t1w.shape}")
    # print(f"fieldmap. {fieldmap.shape}")

    # print("BEFORE converting to torch")
    # Convert to torch format
    t1w = data_util.nii2torch(t1w)
    b0d = data_util.nii2torch(b0d)
    b0u = data_util.niiu2torch(b0u)
    b0_mask = data_util.niimask2torch(b0_mask, repetitions=number_timesteps) != 0
    fieldmap = data_util.niiu2torch(fieldmap)
    # print("AFTER converting to torch")

    # Data normalization
    t1w = data_util.normalize_img(t1w, 150, 0, 1, -1)  # Based on freesurfers T1 normalization
    max_b0d = np.percentile(b0d, 99)  # This usually makes majority of CSF be the upper bound
    min_b0d = 0  # Assumes lower bound is zero (direct from scanner)
    b0d = data_util.normalize_img(b0d, max_b0d, min_b0d, 1, -1)
    b0u = data_util.normalize_img(b0u, max_b0d, min_b0d, 1, -1)  # Use min() and max() from distorted data

    b0_mask = np.array(b0_mask, dtype=np.uint8)

    # Retrieve affines
    b0d_affine = nib.load(b0d_path).affine
    b0d_affine = np.repeat(b0d_affine[None, :], number_timesteps, axis=0)
    fieldmap_affine = nib.load(fieldmap_path).affine
    fieldmap_affine = np.repeat(fieldmap_affine[None, :], number_timesteps, axis=0)

    return (t1w, b0d, b0u, b0_mask, fieldmap, b0d_affine, fieldmap_affine, echospacing, phaseencodingdirection)


def load_data_from_path_for_train(subject_path, use_cache=True, use_saved_nifti=True):
    cached_dir = os.path.join(subject_path, "cache")
    cached_T1 = os.path.join(cached_dir, "processed_T1w.nii.gz")
    cached_b0 = os.path.join(cached_dir, "processed_b0_d_10.nii.gz")
    cached_fieldmap = os.path.join(cached_dir, "processed_fieldmap.nii.gz")

    if use_saved_nifti and os.path.exists(cached_T1) and os.path.exists(cached_b0) and os.path.exists(cached_fieldmap):
        # print(f"Loading preprocessed NIFTI files for subject: {subject_path}")
        img_t1 = nib.load(cached_T1).get_fdata()
        img_b0_d_10 = nib.load(cached_b0).get_fdata()
        img_fieldmap = nib.load(cached_fieldmap).get_fdata()

        # img_t1 = torch.from_numpy(img_t1)
        # img_b0_d_10 = torch.from_numpy(img_b0_d_10)
        # img_fieldmap = torch.from_numpy(img_fieldmap)

        return (img_t1, img_b0_d_10, img_fieldmap)
    
    # print(f"Cached NIFTI files not found for subject: {subject_path}. Processing raw data...")

    t1_path = os.path.join(subject_path, "T1w.nii.gz")
    b0_d_10_path = os.path.join(subject_path, "b0_d_10.nii.gz")
    fieldmap_path = os.path.join(subject_path, "field_map.nii.gz")

    dataset_path = Path(subject_path).parent.absolute()
    with open(os.path.join(dataset_path, 'dataset_meta.json')) as f:
        dataset_meta = json.load(f)

    img_t1 = data_util.get_nii_img(t1_path)
    img_b0_d_10 = data_util.get_nii_img(b0_d_10_path)
    img_fieldmap = data_util.get_nii_img(fieldmap_path)[:, :, :, 0]
    number_timesteps = img_b0_d_10.shape[3]

    if len(img_t1.shape) == 3:
        img_t1 = np.repeat(img_t1[None, :], number_timesteps, axis=0)
        img_t1 = np.transpose(img_t1, axes=(1, 2, 3, 0))

    if len(img_fieldmap.shape) == 3:
        img_fieldmap = np.repeat(img_fieldmap[None, :], number_timesteps, axis=0)
        img_fieldmap = np.transpose(img_fieldmap, axes=(1, 2, 3, 0))

    # Convert to torch img format
    img_t1 = data_util.nii2torch(img_t1)
    img_b0_d_10 = data_util.nii2torch(img_b0_d_10)
    img_fieldmap = data_util.niiu2torch(img_fieldmap)

    # Normalize data
    img_t1 = data_util.normalize_img(img_t1, 150, 0, 1, -1)
    max_img_b0_d = np.percentile(img_b0_d_10, 99)
    min_img_b0_d = 0
    img_b0_d_10 = data_util.normalize_img(img_b0_d_10, max_img_b0_d, min_img_b0_d, 1, -1)

    # # Retrieving affines
    # fieldmap_affine = nib.load(fieldmap_path).affine
    # fieldmap_affine = np.repeat(fieldmap_affine[None, :], number_timesteps, axis=0)
    # echo_spacing = np.array(dataset_meta['echospacing'])
    # echo_spacing = np.repeat(echo_spacing, number_timesteps, axis=0)
    # unwarp_direction = dataset_meta["phaseencodingdirection"]
    # unwarp_direction = np.array([unwarp_direction] * number_timesteps)

    # Caching the data
    os.makedirs(cached_dir, exist_ok=True)
    t1_nii = nib.Nifti1Image(img_t1, nib.load(t1_path).affine)
    b0_nii = nib.Nifti1Image(img_b0_d_10, nib.load(b0_d_10_path).affine)
    fieldmap_nii = nib.Nifti1Image(img_fieldmap, nib.load(fieldmap_path).affine)
    nib.save(t1_nii, cached_T1)
    nib.save(b0_nii, cached_b0)
    nib.save(fieldmap_nii, cached_fieldmap)

    print(f"Processed NIFTI files aved for subject: {subject_path} in {cached_dir}!")

    return (img_t1, img_b0_d_10, img_fieldmap)

# def load_data_from_path_for_train(subject_path):
#     # Collect paths
#     t1_path = os.path.join(subject_path, "T1w.nii.gz")
#     b0_d_10_path = os.path.join(subject_path, "b0_d_10.nii.gz")
#     fieldmap_path = os.path.join(subject_path, "field_map.nii.gz")

#     # Retrieve dataset path
#     # dataset_path = Path(subject_path).parent.absolute()
#     # with open(os.path.join(dataset_path, 'dataset_meta.json')) as f:
#     #     dataset_meta = json.load(f)

#     # Get image
#     img_t1 = data_util.get_nii_img(t1_path)
#     img_b0_d_10 = data_util.get_nii_img(b0_d_10_path)
#     img_fieldmap = data_util.get_nii_img(fieldmap_path)[:, :, :, 0]

#     # Calculate timesteps and extend
#     number_timesteps = img_b0_d_10.shape[3]

#     # Repeat T1 image if we only have one to mach the number of timesteps for training
#     if len(img_t1.shape) == 3:
#         img_t1 = np.repeat(img_t1[None, :], number_timesteps, axis=0)
#         img_t1 = np.transpose(img_t1, axes=(1, 2, 3, 0))

#     if len(img_fieldmap.shape) == 3:
#         img_fieldmap = np.repeat(img_fieldmap[None, :], number_timesteps, axis=0)
#         img_fieldmap = np.transpose(img_fieldmap, axes=(1, 2, 3, 0))

#     # Convert to torch img format
#     img_t1 = data_util.nii2torch(img_t1)
#     img_b0_d_10 = data_util.nii2torch(img_b0_d_10)
#     img_fieldmap = data_util.niiu2torch(img_fieldmap)

#     # Normalize data
#     img_t1 = data_util.normalize_img(img_t1, 150, 0, 1, -1)  # Based on freesurfers T1 normalization
#     max_img_b0_d = np.percentile(img_b0_d_10, 99)  # This usually makes majority of CSF be the upper bound
#     min_img_b0_d = 0  # Assumes lower bound is zero (direct from scanner)
#     img_b0_d_10 = data_util.normalize_img(img_b0_d_10, max_img_b0_d, min_img_b0_d, 1, -1)

#     # fieldmap_affine = nib.load(fieldmap_path).affine
#     # fieldmap_affine = np.repeat(fieldmap_affine[None, :], number_timesteps, axis=0)

#     # echo_spacing = np.array(dataset_meta['echospacing'])
#     # echo_spacing = np.repeat(echo_spacing, number_timesteps, axis=0)

#     # unwarp_direction = dataset_meta["phaseencodingdirection"]
#     # unwarp_direction = np.array([unwarp_direction] * number_timesteps)
#     # unwarp_direction = np.repeat(unwarp_direction, number_timesteps, axis=0)

#     # Return the loaded images along with the test images if available
#     return (img_t1, img_b0_d_10, img_fieldmap) # , fieldmap_affine, echo_spacing, unwarp_direction)



def load_data_from_path(subject_path):
    """
    Function for retrieving a particular subject data from a path
    """
    # Get paths
    t1_path = os.path.join(subject_path, 'T1w.nii.gz')
    b0_d_path = os.path.join(subject_path, 'b0_d.nii.gz')
    b0_u_path = os.path.join(subject_path, 'b0_u.nii.gz')
    mask_path = os.path.join(subject_path, 'b0_mask.nii.gz')
    fieldmap_path = os.path.join(subject_path, 'field_map.nii.gz')

    # Define test file paths
    b0alltf_d_path = os.path.join(subject_path, 'b0alltf_d.nii.gz')
    b0alltf_u_path = os.path.join(subject_path, 'b0alltf_u.nii.gz')

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
    img_fieldmap_affine = data_util.load_only_nii(fieldmap_path).affine
    img_b0_d_affine = data_util.load_only_nii(b0_d_path).affine

    # Check if the test files exists and act accordingly
    if os.path.exists(b0alltf_d_path) and os.path.exists(b0alltf_u_path):
        img_b0alltf_d = data_util.get_nii_img(b0alltf_d_path)
        img_b0alltf_u = data_util.get_nii_img(b0alltf_u_path)
    else:
        img_b0alltf_d = None
        img_b0alltf_u = None

    '''# Pad array since I stupidly used template with dimensions not factorable by 8
    # Assumes input is (77, 91, 77) and pad to (80, 96, 80) with zeros
    img_t1 = np.pad(img_t1, ((2, 1), (3, 2), (2, 1), (0, 0)), 'constant')
    img_b0_d = np.pad(img_b0_d, ((2, 1), (3, 2), (2, 1), (0, 0)), 'constant')
    img_b0_u = np.pad(img_b0_u, ((2, 1), (3, 2), (2, 1), (0, 0)), 'constant')
    img_mask = np.pad(img_mask, ((2, 1), (3, 2), (2, 1), (0, 0)), 'constant')'''

    number_timesteps = img_b0_d.shape[3]

    # Repeat T1 image if we only have one to mach the number of timesteps for training
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
    
    # Check for all tf
    if img_b0alltf_d is not None and img_b0alltf_u is not None:
        img_b0alltf_d = data_util.nii2torch(img_b0alltf_d)
        img_b0alltf_u = data_util.niiu2torch(img_b0alltf_u)
        # Optionally normalize these images if they follow similar conventions
        img_b0alltf_d = data_util.normalize_img(img_b0alltf_d, max_img_b0_d, min_img_b0_d, 1, -1)
        img_b0alltf_u = data_util.normalize_img(img_b0alltf_u, max_img_b0_d, min_img_b0_d, 1, -1)

    img_mask = np.array(img_mask, dtype=np.uint8)

    b0u_affine = nib.load(b0_u_path).affine
    b0u_affine = np.repeat(b0u_affine[None, :], number_timesteps, axis=0)

    b0d_affine = nib.load(b0_d_path).affine
    b0d_affine = np.repeat(b0d_affine[None, :], number_timesteps, axis=0)

    fieldmap_affine = nib.load(fieldmap_path).affine
    fieldmap_affine = np.repeat(fieldmap_affine[None, :], number_timesteps, axis=0)

    echo_spacing = np.array(dataset_meta['echospacing'])
    echo_spacing = np.repeat(echo_spacing, number_timesteps, axis=0)

    # Return the loaded images along with the test images if available
    return (img_t1, img_b0_d, img_b0_u, img_mask, img_fieldmap, 
            b0u_affine, b0d_affine, fieldmap_affine, echo_spacing,
            img_b0alltf_d, img_b0alltf_u)
