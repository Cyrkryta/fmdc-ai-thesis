import glob
from os import path

import numpy as np
import torch

from project.data import fmri_data_util, data_util
from project.metrics_scripts.metrics_computation_base import MetricsComputationBase
from project.models.unet3d_fieldmap import UNet3DFieldmap


class BaselineModelMetricsComputation(MetricsComputationBase):
    def __init__(self, subject_paths, device):
        super().__init__()

        self.subject_paths = subject_paths
        self.device = device

    def get_subject_paths(self):
        return list(glob.glob(self.subject_paths))

    def _load_all_data(self, input_path, output_path, timestep_idx):
        # Get paths
        b0_d_path = path.join(input_path, 'b0.nii.gz')
        b0_u_path = path.join(input_path, 'b0_u.nii.gz')
        mask_path = path.join(input_path, 'b0_mask.nii.gz')
        out_path = path.join(output_path, 'b0_all.nii.gz')

        # Load affine from b0_u using nibabel
        img_b0_u_nii = data_util.load_only_nii(b0_u_path)
        affine_b0_u = img_b0_u_nii.affine

        # Get image
        img_b0_d = data_util.get_nii_img(b0_d_path)
        img_b0_u = data_util.get_nii_img(b0_u_path)[:, :, :, timestep_idx]
        # img_b0_u = data_util.get_nii_img(b0_u_path) # TODO: Check if this is correct
        img_mask = data_util.get_nii_img(mask_path)
        img_out = data_util.get_nii_img(out_path)[:, :, :, 0]
        # img_out = data_util.get_nii_img(out_path) # TODO: Check if this is correct

        img_b0_d = np.transpose(img_b0_d, axes=(2, 0, 1))
        img_b0_u = np.transpose(img_b0_u, axes=(2, 0, 1))
        img_mask = np.transpose(img_mask, axes=(2, 0, 1))
        img_out = np.transpose(img_out, axes=(2, 0, 1))

        img_b0_u = np.expand_dims(img_b0_u, axis=0)
        img_mask = np.expand_dims(img_mask, axis=0)

        # Normalize data
        max_img_b0_d = np.percentile(img_b0_d, 99)  # This usually makes majority of CSF be the upper bound
        min_img_b0_d = 0  # Assumes lower bound is zero (direct from scanner)
        img_b0_d = data_util.normalize_img(img_b0_d, max_img_b0_d, min_img_b0_d, 1, -1)
        img_b0_u = data_util.normalize_img(img_b0_u, max_img_b0_d, min_img_b0_d, 1,
                                           -1)  # Use min() and max() from distorted data
        img_out = data_util.normalize_img(img_out, max_img_b0_d, min_img_b0_d, 1,
                                           -1)  # Use min() and max() from distorted data

        img_mask = np.array(img_mask != 0, dtype=np.uint8)

        return img_b0_d, img_b0_u, img_mask, img_out, affine_b0_u
    
    def load_input_samples(self, subject_path):
        time_series = []

        for idx, time_step_path in enumerate(sorted(glob.glob(path.join(subject_path, 't-*')))):
            img_b0_d, img_b0_u, img_mask, img_out, affine_b0_out = self._load_all_data(
                input_path=time_step_path,
                output_path=time_step_path.replace('INPUTS', 'OUTPUTS'),
                timestep_idx=idx
            )

            time_series.append({
                'b0d': img_b0_d,
                'b0u': img_b0_u,
                'mask': img_mask,
                'model_output': img_out,
                'b0u_affine': affine_b0_out,
                'fieldmap_affine': None # The baseline model doesn't create any fieldmap
            })

        return time_series

    def get_undistorted_b0(self, sample):
        return sample['model_output'], None