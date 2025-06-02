import glob
import json
import os
import tempfile
from os import path
from os.path import abspath
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

import nipype.interfaces.io as nio
from nipype.interfaces import fsl
from nipype import SelectFiles, Node, Function, Workflow

import subprocess
import time

from project.data import data_util
from project.metrics_scripts.metrics_computation_base import MetricsComputationBase


class MeanFieldmapsMetricsComputation(MetricsComputationBase):
    def __init__(self, subject_paths, device):
        super().__init__()

        self.subject_paths = subject_paths
        self.device = device
        self.undistortion_compute_times = []

    def get_subject_paths(self):
        return self.subject_paths
        return list(glob.glob(self.subject_paths))

    def _load_data_from_path(self, subject_path):
        # Get paths
        t1_path = os.path.join(subject_path, 'T1w.nii.gz')
        b0_d_path = os.path.join(subject_path, 'b0_d.nii.gz')
        b0_u_path = os.path.join(subject_path, 'b0_u.nii.gz')
        mask_path = os.path.join(subject_path, 'b0_mask.nii.gz')
        mean_fieldmap_path = os.path.join(subject_path, 'mean_fieldmap.nii.gz')

        # Get meta information
        dataset_path = Path(subject_path).parent.absolute()
        with open(os.path.join(dataset_path, 'dataset_meta.json')) as f:
            dataset_meta = json.load(f)

        # Load meta data
        echospacing = dataset_meta["echospacing"]
        unwarp_direction = dataset_meta["phaseencodingdirection"]
        

        # Get image
        img_t1 = data_util.get_nii_img(t1_path)
        img_b0_d = data_util.get_nii_img(b0_d_path)
        img_b0_u = data_util.get_nii_img(b0_u_path)
        img_mask = data_util.get_nii_img(mask_path)
        img_mean_fieldmap = data_util.get_nii_img(mean_fieldmap_path)

        number_timesteps = img_b0_d.shape[3]

        # Repeat T1 image if we only have one
        if len(img_t1.shape) == 3:
            img_t1 = np.repeat(img_t1[None, :], number_timesteps, axis=0)
            img_t1 = np.transpose(img_t1, axes=(1, 2, 3, 0))

        if len(img_mean_fieldmap.shape) == 3:
            img_fieldmap = np.repeat(img_mean_fieldmap[None, :], number_timesteps, axis=0)
            img_fieldmap = np.transpose(img_fieldmap, axes=(1, 2, 3, 0))

        # Convert to torch img format
        img_t1 = data_util.nii2torch(img_t1)
        img_b0_d = data_util.nii2torch(img_b0_d)
        img_b0_u = data_util.niiu2torch(img_b0_u)
        img_mask = data_util.niimask2torch(img_mask, repetitions=number_timesteps) != 0
        img_fieldmap = data_util.niiu2torch(img_fieldmap)

        # Normalize data
        img_t1 = data_util.normalize_img(img_t1, 150, 0, 1, -1) 
        max_img_b0_d = np.percentile(img_b0_d, 99)
        min_img_b0_d = 0 
        img_b0_d = data_util.normalize_img(img_b0_d, max_img_b0_d, min_img_b0_d, 1, -1)
        img_b0_u = data_util.normalize_img(img_b0_u, max_img_b0_d, min_img_b0_d, 1,
                                           -1) 

        img_mask = np.array(img_mask, dtype=np.uint8)

        b0u_affine = nib.load(b0_u_path).affine
        b0u_affine = np.repeat(b0u_affine[None, :], number_timesteps, axis=0)

        fieldmap_affine = nib.load(mean_fieldmap_path).affine
        fieldmap_affine = np.repeat(fieldmap_affine[None, :], number_timesteps, axis=0)

        echo_spacing = np.array(dataset_meta['echoSpacing'])
        echo_spacing = np.repeat(echo_spacing, number_timesteps, axis=0)

        return img_t1, img_b0_d, img_b0_u, img_mask, img_fieldmap, b0u_affine, fieldmap_affine, echo_spacing
    
    def _apply_fugue(self, BOLDd_path, mfm_path, echo_spacing, unwarp_direction, out_path):
        custom_tmpdir = "/home/mlc/dev/fmdc/fmdc-ai-thesis/project/metrics_scripts/tmpdir"
        with tempfile.TemporaryDirectory(dir=custom_tmpdir) as tmpdir:
            cmd = [
                "fugue",
                "-i", BOLDd_path,
                "--loadfmap=" + mfm_path,
                "--dwell=" + str(echo_spacing),
                "--smooth3=3",
                "--unwarpdir=" + unwarp_direction,
                "-u", out_path
            ]
            subprocess.run(cmd, check=True)

    def load_input_samples(self, subject_path):
        # Find paths
        t1_path = os.path.join(subject_path, 'T1w.nii.gz')      # T1
        b0_d_path = os.path.join(subject_path, 'b0_d.nii.gz')   # Distorted
        b0_u_path = os.path.join(subject_path, 'b0_u.nii.gz')   # GT undistorted
        mask_path = os.path.join(subject_path, 'b0_mask.nii.gz')
        mean_fieldmap_path = os.path.join(subject_path, 'mean_fieldmap.nii.gz')

        # Get meta information
        dataset_path = Path(subject_path).parent.absolute()
        with open(os.path.join(dataset_path, 'dataset_meta.json')) as f:
            dataset_meta = json.load(f)

        # Load meta data
        echospacing = dataset_meta["echospacing"]
        unwarp_direction = dataset_meta["phaseencodingdirection"]

        # Performing fugue
        fugue_out_path = os.path.join(subject_path, "BOLD_mfm_fugue.nii.gz")

        print(f"\nApplying fugue...")
        undistort_start = time.time()
        self._apply_fugue(
            BOLDd_path=b0_d_path,
            mfm_path=mean_fieldmap_path,
            echo_spacing=echospacing,
            unwarp_direction=unwarp_direction,
            out_path=fugue_out_path
        )
        undistort_end = time.time()
        print("fugue applied...\n")
        undistort_elapsed = undistort_end - undistort_start
        self.undistortion_compute_times.append(undistort_elapsed)

        print(f"Loading fugue corrected image...")
        BOLD_u = nib.load(fugue_out_path).get_fdata()
        print(f"Number of timepoints after unwarp: {BOLD_u.shape[3]}")

        print(f"\nLoading images for comparison...")
        t1_img = nib.load(t1_path).get_fdata()
        BOLD_d = nib.load(b0_d_path).get_fdata()
        BOLD_u_gt = nib.load(b0_u_path).get_fdata()
        BOLD_mask = nib.load(mask_path).get_fdata()
        print(f"Comparison images loaded...")

        print(f"\nNormalizing data...")
        t1_img = data_util.normalize_img(t1_img, 150, 0, 1, -1)
        max_img_BOLDd = np.percentile(BOLD_d, 99)
        min_img_BOLDd = 0
        BOLD_d = data_util.normalize_img(BOLD_d, max_img_BOLDd, min_img_BOLDd, 1, -1)
        BOLD_u_gt = data_util.normalize_img(BOLD_u_gt, max_img_BOLDd, min_img_BOLDd, 1, -1) 
        BOLD_u = data_util.normalize_img(BOLD_u, max_img_BOLDd, min_img_BOLDd, 1, -1)
        # Evaluate the time series
        time_series = []
        num_timepoints = BOLD_u.shape[3]
        print(f"\nFound {num_timepoints} time points...")

        # Check if there is a dimension mismatch
        assert BOLD_u.shape[3] == BOLD_u_gt.shape[3], \
            f"Mismatch: {BOLD_u.shape[3]} time points found but expected {BOLD_u_gt.shape[3]}"
        
        # Go through the time points
        for t in range(BOLD_u_gt.shape[3]):
            b0d = BOLD_d[..., t]
            b0u = BOLD_u_gt[..., t]
            out = BOLD_u[..., t]
            mask = BOLD_mask

            sample = {
                "b0d": b0d,
                "b0u": b0u,
                "out": out,
                "mask": mask
            }

            time_series.append(sample)
        
        print(f"Finished loading time points...")
        return time_series

    def get_undistorted_b0(self, sample):
        return sample["out"], None
    
    def save_compute_times(self, save_path):
        with open(save_path, "w") as f:
            f.write("Total Compute Times (seconds): \n")
            f.write(", ".join([f"{t:.3f}" for t in self.undistortion_compute_times]) + "\n")
        print(f"Compute times saved to {save_path}")
