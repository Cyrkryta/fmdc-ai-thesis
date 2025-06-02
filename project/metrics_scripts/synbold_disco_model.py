import os
import nibabel as nib
import numpy as np
from project.metrics_scripts.metrics_computation_base import MetricsComputationBase
from project.data import data_util
from skimage import metrics

class SynboldDiscoModelMetricsComputation(MetricsComputationBase):
    def __init__(self, inputs_dir, outputs_dir, subject_paths, device):
        super().__init__()
        self.inputs_dir = inputs_dir
        self.outputs_dir = outputs_dir
        self.subject_paths = subject_paths
        self.device = device

    def get_subject_paths(self):
        return self.subject_paths
    
    def _load_all_data(self, input_path, output_path, timestep_idx):
        b0_d_path = os.path.join(input_path, "BOLD_d.nii.gz")
        b0_u_path = os.path.join(input_path, "b0_u.nii.gz")
        mask_path = os.path.join(input_path, "b0_mask.nii.gz")
        out_path = os.path.join(output_path, "BOLD_u.nii.gz")

        img_b0_d_full = data_util.get_nii_img(b0_d_path)
        img_b0_u_full = data_util.get_nii_img(b0_u_path)
        img_out_full = data_util.get_nii_img(out_path)
        img_mask = data_util.get_nii_img(mask_path)

        img_b0_u_nii = data_util.load_only_nii(b0_u_path)
        affine_b0_u = img_b0_u_nii.affine

        print(f"Distorted shape before time split: {img_b0_d_full.shape}")
        print(f"Undistorted shape before time split: {img_b0_u_full.shape}")
        print(f"Output shape before time split: {img_out_full.shape}")

        img_b0_d = img_b0_d_full[:, :, :, timestep_idx]
        img_b0_u = img_b0_u_full[:, :, :, timestep_idx]
        img_out = img_out_full[:, :, :, timestep_idx]

        img_b0_d = np.transpose(img_b0_d, axes=(2, 0, 1))
        img_b0_u = np.transpose(img_b0_u, axes=(2, 0, 1))
        img_mask = np.transpose(img_mask, axes=(2, 0, 1))
        img_out = np.transpose(img_out, axes=(2, 0, 1))

        img_b0_u = np.expand_dims(img_b0_u, axis=0)
        img_mask = np.expand_dims(img_mask, axis=0)

        max_img_b0_d = np.percentile(img_b0_d, 99)
        min_img_b0_d = 0
        img_b0_d = data_util.normalize_img(img_b0_d, max_img_b0_d, min_img_b0_d, 1, -1)
        img_b0_u = data_util.normalize_img(img_b0_u, max_img_b0_d, min_img_b0_d, 1, -1)
        img_out = data_util.normalize_img(img_out, max_img_b0_d, min_img_b0_d, 1, -1)
        img_mask = np.array(img_mask != 0, dtype=np.uint8)

        return img_b0_d, img_b0_u, img_mask, img_out, affine_b0_u
    
    def load_input_samples(self, subject_path):
        input_subject_path = os.path.join(self.inputs_dir, subject_path)
        output_subject_path = os.path.join(self.outputs_dir, subject_path)
        print(f"Processing inputs subject: {input_subject_path}")
        print(f"Output subject paht: {output_subject_path}\n")

        img_b0_u_full = nib.load(os.path.join(input_subject_path, "b0_u.nii.gz")).get_fdata()
        num_timepoints = img_b0_u_full.shape[3]
        print(f"Number of timepoints: {num_timepoints}")
        time_series = []
        for t in range(num_timepoints):
            print(f"Loading timepoint {t} for subject {subject_path}")
            img_b0_d, img_b0_u, img_mask, img_out, affine_b0_out = self._load_all_data(
                input_subject_path, output_subject_path, t
            )

            print(f"Distorted shape after loading: {img_b0_d.shape}")
            print(f"Undistorted shape after loading: {img_b0_u.shape}")
            print(f"Mask shape after loading: {img_mask.shape}")
            print(f"Output image after loading: {img_out.shape}")
            
            sample = {
                "b0u": img_b0_u,            
                "b0d": img_b0_d,           
                "mask": img_mask,           
                "model_output": img_out,    
                "b0u_affine": affine_b0_out,
                "fieldmap_affine": None 
            }
            time_series.append(sample)
            print(f"Loaded timepoint {t} for subject {subject_path}")
        return time_series
    
    def get_undistorted_b0(self, sample):
        return sample["model_output"], None
