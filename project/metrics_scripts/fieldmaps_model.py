# Import all of the dependenceis
import glob
import json
import numpy as np
import torch
from project.data import fmri_data_util, data_util
from project.metrics_scripts.metrics_computation_base import MetricsComputationBase
from project.models.unet3d_fieldmap import UNet3DFieldmap
import tempfile
import os
import nibabel as nib
import subprocess
import time

"""
Class:
Computation base for computing the metrics on the fieldmap model
"""
class FieldmapsModelMetricsComputation(MetricsComputationBase):
    def __init__(self, CHECKPOINT_PATH, TEST_PATHS, device):
        # Call super
        super().__init__()
        # Define class variables and load the model
        self.TEST_PATHS = TEST_PATHS
        self.device = device
        self.model = UNet3DFieldmap.load_from_checkpoint(CHECKPOINT_PATH, map_location=torch.device(device), encoder_map_location=torch.device(device), device=device)
        self.model.to(device)
        self.model.eval()
        self.undistortion_compute_times = []
        self.fieldmap_compute_times = []

    # Get all of the subject paths (computing the data again - should be changed)
    def get_subject_paths(self):
        return self.TEST_PATHS
    
    # Fieldmap prediction
    def _predict_fieldmap(self, bold_ref, t1_ref):
        print(f"Input dimensions: {bold_ref.shape}, {t1_ref.shape}")
        input_img = np.stack((bold_ref, t1_ref))
        print(f"Input image shape: {input_img.shape}")
        input_tensor = torch.from_numpy(input_img).unsqueeze(0).float().to(self.device)
        print(f"Input tensor of type: {type(input_tensor)}")
        with torch.no_grad():
            pred = self.model(input_tensor)[0].detach().cpu().numpy()
        print(f"Pred type: {type(pred)}, dimension: {pred.shape}")
        fieldmap_pred = pred[0]
        fieldmap_nifti = np.transpose(fieldmap_pred, (1, 2, 0))
        print(f"Final fieldmap shape: {fieldmap_nifti.shape}")
        return fieldmap_nifti

    # FUGUE undistortion
    def _apply_fugue_undistortion(self, bold_4d, pred_fieldmap, affine, echo_spacing, unwarp_direction, out_path):
        custom_tmpdir = "/indirect/student/magnuschristensen/dev/fmdc/fmdc-ai-thesis/project/metrics_scripts/tmpdir"
        with tempfile.TemporaryDirectory(dir=custom_tmpdir) as tmpdir:
            bold_path = os.path.join(tmpdir, "bold.nii.gz")
            fmap_path = os.path.join(tmpdir, "fmap.nii.gz")
            print(bold_4d.shape, pred_fieldmap.shape)
            nib.save(nib.Nifti1Image(bold_4d, affine), bold_path)
            nib.save(nib.Nifti1Image(pred_fieldmap, affine), fmap_path)
            
            # Build the FUGUE command
            cmd = [
                "fugue",
                "-i", bold_path,
                "--loadfmap=" + fmap_path,
                "--dwell=" + str(echo_spacing),
                "--smooth3=3",
                "--unwarpdir=" + unwarp_direction,
                "-u", out_path
            ]

            subprocess.run(cmd, check=True)
            
    # Load the input samples
    def load_input_samples(self, subject_path):
        print(f"Loading inputs samples for subject...")
        # Load all subject data for testing
        (t1w, b0d, b0u, b0_mask, fieldmap, b0d_affine, fieldmap_affine, echospacing, phaseencodingdirection) = fmri_data_util.load_data_from_path_for_test(subject_path=subject_path)
        print(f"Retrieved samples of the following shapes:")
        print(f"T1w shape: {t1w.shape}")
        print(f"b0d shape: {b0d.shape}")
        print(f"b0u shape: {b0u.shape}")
        print(f"mask shape: {b0_mask.shape}")
    
        print(f"Transposing input for the ")
        bold_4d = np.transpose(b0d, (1, 2, 3, 0))
        t1_4d = np.transpose(t1w, (1, 2, 3, 0))
        print(f"BOLD 4D shape after transpose: {bold_4d.shape}")
        print(f"T1 4D shape after transpose: {t1_4d.shape}\n")

        print(f"Undistorting BOLD with fugue...")
        start_time = time.time()
        undistorted_4d = self.model.undistort_full_sequence(bold_4d, t1_4d, b0d_affine, b0d_affine, echospacing, phaseencodingdirection, out_path=os.path.join(subject_path, 'b0_fugue_u.nii.gz'))
        end_time = time.time()
        print(f"Undistortion completed...")
        elapsed_time = end_time - start_time
        self.undistortion_compute_times.append(elapsed_time)

        print(f"Creating samples (Loading time points)....")
        time_series = []
        num_tp = undistorted_4d.shape[3]
        for t in range(num_tp):
            sample = {
                "b0d": np.transpose(list(b0d)[t], axes=(1,2,0)),
                "b0u": np.transpose(list(b0u)[t].squeeze(0), axes=(1,2,0)),
                "out": undistorted_4d[..., t],
                "mask": np.transpose(list(b0_mask)[t].squeeze(0), axes=(1,2,0))
            }
            time_series.append(sample)
        print(f"Samples loaded")
        return time_series

        # print(f"Original t1 shape: {t1w.shape}")
        # print(f"BOLD d original shape: {b0d.shape}")
        bold_4d = np.transpose(b0d, (1, 2, 3, 0))
        t1_trans = np.transpose(t1w, (1, 2, 3, 0))
        num_timepoints_orig = bold_4d.shape[3]
        print(f"Number of timepoints found (Distorted): {num_timepoints_orig}")
        mid_index = bold_4d.shape[3] // 2
        bold_ref = bold_4d[..., mid_index]
        t1_ref = t1_trans[..., mid_index]
        print(type(bold_4d), type(t1_trans))
        print(bold_ref.shape, t1_ref.shape)


        # START TIME
        undistort_start = time.time()
        fieldmap_start = time.time()
        pred_fieldmap = self._predict_fieldmap(bold_ref=bold_ref, t1_ref=t1_ref)
        fieldmap_end = time.time()
        fieldmap_elapsed = fieldmap_end - fieldmap_start
        affine_used = b0d_affine[0]
        out_bold_path = os.path.join(subject_path, "b0_fugue_u.nii.gz")
        print(f"Applying fugue")
        bold_4d_fugue = np.transpose(bold_4d, axes=(1, 2, 0, 3))
        self._apply_fugue_undistortion(
            bold_4d=bold_4d_fugue,
            pred_fieldmap=pred_fieldmap,
            affine=affine_used,
            echo_spacing=echospacing,
            unwarp_direction=phaseencodingdirection,
            out_path=out_bold_path
        )
        undistort_end = time.time()
        undistort_elapsed = undistort_end - undistort_start 

        self.fieldmap_compute_times.append(fieldmap_elapsed)
        self.undistortion_compute_times.append(undistort_elapsed)

        unwarped_4d = nib.load(out_bold_path).get_fdata()
        print(f"Number of timepoints after unwarp: {unwarped_4d.shape[3]}")

        # Load the unwarped 4D BOLD volume
        unwarped_4d = nib.load(out_bold_path).get_fdata()
        print(f"Number of timepoints after unwarp: {unwarped_4d.shape[3]}")

        time_series = []
        num_timepoints=unwarped_4d.shape[3]
        print(f"Found {num_timepoints} time points")

        assert len(list(t1w)) == unwarped_4d.shape[3], \
            f"Mismatch: {len(list(t1w))} T1 timepoints vs. {unwarped_4d.shape[3]} undistorted volumes"

        time_series = []
        for t in range(len(list(t1w))):
            # print(f"Loading timepoint {t}")
            img_b0d = list(b0d)[t]
            img_b0u = list(b0u)[t]
            img_out = unwarped_4d[..., t]
            img_mask = list(b0_mask)[t]

            sample = {
                "b0d": np.transpose(img_b0d, axes=(1, 2, 0)),
                "b0u": np.transpose(img_b0u.squeeze(0), axes=(1, 2, 0)),
                "out": img_out,
                "mask": np.transpose(img_mask.squeeze(0), axes=(1, 2, 0))
            }

            time_series.append(sample)
            # print(f"Loaded timepoint {t}...\n")
        
        print(f"Finished loading time points")
        return time_series

    def get_undistorted_b0(self, sample):
        return sample["out"], None
    
    def save_compute_times(self, save_path):
        with open(save_path, "w") as f:
            f.write("Undistortion Times (seconds):")
            if self.undistortion_compute_times:
                f.write(", ".join(f"{t:.3f}" for t in self.undistortion_compute_times) + "")
            else:
                f.write("No undistortion time recorded...")
        print(f"Compute times saved to {save_path}")

    
    # def save_compute_times(self, save_path):
    #     with open(save_path, "w") as f:
    #         f.write("Fieldmap Prediction Times (seconds):\n")
    #         f.write(", ".join([f"{t:.3f}" for t in self.fieldmap_compute_times]) + "\n")
    #         f.write("Total Compute Times (seconds): \n")
    #         f.write(", ".join([f"{t:.3f}" for t in self.undistortion_compute_times]) + "\n")
    #     print(f"Compute times saved to {save_path}")
        



    #     bold_vols = [np.array(vol) for vol in b0d]
    #     bold_4d = np.stack(bold_vols)
    #     bold_4d = np.transpose(bold_4d, (1, 2, 3, 0))
    #     num_timepoints_orig = bold_4d.shape[3]
    #     print(f"Number of timepoints found (Distorted): {num_timepoints_orig}")
    #     mid_index = bold_4d.shape[3] // 2
    #     bold_ref = bold_4d[..., mid_index] # Take the middle timepoint
    #     t1_ref = t1w[..., t1w.shape[3] // 2] if t1w.ndim == 4 else np.array(t1w)
    #     pred_fieldmap = self._predict_fieldmap(bold_ref=bold_ref, t1_ref=t1_ref)
    #     affine_used = b0d_affine[0]

    #     # Apply fugue correction
    #     out_bold_path = os.path.join(subject_path, "b0_fugue_u.nii.gz")
    #     self._apply_fugue_undistortion(
    #         bold_4d=bold_4d,
    #         pred_fieldmap=pred_fieldmap,
    #         affine=affine_used,
    #         echo_spacing=echospacing,
    #         unwarp_direction=phaseencodingdirection,
    #         out_path=out_bold_path
    #     )

    #     # Load the unwarped 4D BOLD volume
    #     unwarped_4d = nib.load(out_bold_path).get_fdata()
    #     print(f"Number of timepoints after unwarp: {unwarped_4d.shape[3]}")

    #     time_series = []
    #     num_timepoints=unwarped_4d.shape[3]
    #     print(f"Found {num_timepoints} time points")

    #     assert len(list(t1w)) == unwarped_4d.shape[3], \
    #         f"Mismatch: {len(list(t1w))} T1 timepoints vs. {unwarped_4d.shape[3]} undistorted volumes"

    #     time_series = []
    #     for t in range(len(list(t1w))):
    #         print(f"Loading timepoint {t} for subject {subject_path}...")
    #         img_b0d = list(b0d)[t]
    #         img_b0u = list(b0u)[t]
    #         img_out = unwarped_4d[..., t]
    #         img_mask = list(b0_mask)[t]

    #         sample = {
    #             "b0d": img_b0d,
    #             "b0u": img_b0u,
    #             "out": img_out,
    #             "mask": img_mask
    #         }

    #         time_series.append(sample)
    #         print(f"Loaded timepoint {t}...\n")

    #     return time_series

    #     # # placeholder
    #     # time_series = []

    #     # # Go through each of the timesteps data
    #     # for t_step in range(len(list(t1w))):
    #     #     img_t1w = list(t1w)[t_step]
    #     #     img_b0d = list(b0d)[t_step]
    #     #     img_b0u = list(b0u)[t_step]
    #     #     img_b0mask = list(b0_mask)[t_step]
    #     #     img_data = np.stack((img_b0d, img_t1w))
    #     #     img_fieldmap = list(fieldmap)[t_step]
    #     #     b0d_affine_list = list(b0d_affine)[t_step]
    #     #     fieldmap_affine_list = list(fieldmap_affine)[t_step]
    #     #     echospacing_val = echospacing
    #     #     phaseencodingdirection_val = phaseencodingdirection

    #     #     # Append to the time_series
    #     #     time_series.append({
    #     #         'img': img_data,
    #     #         't1': img_t1w,
    #     #         'b0d': img_b0d,
    #     #         'b0u': img_b0u,
    #     #         'mask': img_b0mask,
    #     #         'fieldmap': img_fieldmap,
    #     #         'b0d_affine': b0d_affine_list,
    #     #         "fieldmap_affine": fieldmap_affine_list,
    #     #         'echo_spacing': echospacing_val,
    #     #         'phase_encoding_direction': phaseencodingdirection_val
    #     #     })

    #     # # Return the timeseries for the subject
    #     # return time_series

    #     # # Load all subject data
    #     # img_t1_all, img_b0_d_all, img_b0_u_all, img_mask_all, img_fieldmap_all, b0u_affine_all, fieldmap_affine_all, echo_spacing_all = fmri_data_util.load_data_from_path(subject_path)

    #     # # Placeholder for the time_series
    #     # time_series = []

    #     # # Go through each of the timesteps
    #     # for time_step in range(len(list(img_t1_all))):
    #     #     # Get the data for the particular timestep
    #     #     img_t1 = list(img_t1_all)[time_step]
    #     #     img_b0_d = list(img_b0_d_all)[time_step]
    #     #     img_b0_u = list(img_b0_u_all)[time_step]
    #     #     img_mask = list(img_mask_all)[time_step]
    #     #     img_data = np.stack((img_b0_d, img_t1))
    #     #     img_fieldmap = list(img_fieldmap_all)[time_step]
    #     #     b0u_affine = list(b0u_affine_all)[time_step]
    #     #     fieldmap_affine = list(fieldmap_affine_all)[time_step]
    #     #     echo_spacing = list(echo_spacing_all)[time_step]

    #     #     # Add the data to the time-series
    #     #     time_series.append({
    #     #         'img': img_data,
    #     #         't1': img_t1,
    #     #         'b0d': img_b0_d,
    #     #         'b0u': img_b0_u,
    #     #         'mask': img_mask,
    #     #         'fieldmap': img_fieldmap,
    #     #         'b0u_affine': b0u_affine,
    #     #         'fieldmap_affine': fieldmap_affine,
    #     #         'echo_spacing': echo_spacing
    #     #     })

    #     # # Return the time-series
    #     # return time_series

    # # Undistorting the image
    # # def get_undistorted_b0(self, sample):
    # #     # Get the data
    # #     input_img = torch.as_tensor(sample['img']).float().to(self.device)
    # #     input_b0d = torch.as_tensor(sample['b0d']).float().to(self.device)
    # #     input_b0d_affine = torch.as_tensor(sample['b0d_affine']).float().to(self.device)
    # #     # input_b0u_affine = torch.as_tensor(sample['b0u_affine']).float().to(self.device)
    # #     input_fieldmap_affine = torch.as_tensor(sample['fieldmap_affine']).float().to(self.device)
    # #     input_echo_spacing = sample["echo_spacing"]
    # #     input_unwarp_direction = sample["phase_encoding_direction"]
    # #     # input_echo_spacing = torch.as_tensor(sample['echo_spacing']).float().to(self.device)
    # #     # input_unwarp_direction = torch.as_tensor(sample['phase_encoding_direction']).float().to(self.device)

    # #     # Estimated output fieldmap
    # #     out = self.model(input_img.unsqueeze(0))

    # #     # Undistorted (unwarped) output
    # #     result_u = self.model._undistort_b0(
    # #         input_b0d,
    # #         out[0],
    # #         input_b0d_affine,
    # #         input_fieldmap_affine,
    # #         input_echo_spacing,
    # #         input_unwarp_direction
    # #     )
    # #     # result = self.model._undistort_b0(input_b0d, out[0], input_b0u_affine, input_fieldmap_affine, input_echo_spacing)

    # #     # Convert to torch and output
    # #     return data_util.nii2torch(result_u)[0], out[0].detach().cpu().numpy()