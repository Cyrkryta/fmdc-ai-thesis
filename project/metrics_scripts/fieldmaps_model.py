# Import all of the dependenceis
import glob
import json
import numpy as np
import torch
from project.data import fmri_data_util, data_util
from project.metrics_scripts.metrics_computation_base import MetricsComputationBase
from project.models.unet3d_fieldmap import UNet3DFieldmap

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

    # Get all of the subject paths (computing the data again - should be changed)
    def get_subject_paths(self):
        return self.TEST_PATHS

        subject_paths = fmri_data_util.collect_all_subject_paths(dataset_paths=glob.glob(self.dataset_root))

        total_count = len(subject_paths)
        train_count = int(0.7 * total_count)
        val_count = int(0.2 * total_count)
        test_count = total_count - train_count - val_count

        rng = torch.Generator()
        rng.manual_seed(0)
        _, val_paths, test_paths = torch.utils.data.random_split(
            subject_paths, (train_count, val_count, test_count),
            generator=rng
        )

        return list(test_paths)

    # Load the input samples
    def load_input_samples(self, subject_path):
        # Load all subject data
        img_t1_all, img_b0_d_all, img_b0_u_all, img_mask_all, img_fieldmap_all, b0u_affine_all, fieldmap_affine_all, echo_spacing_all = fmri_data_util.load_data_from_path(subject_path)

        # Placeholder for the time_series
        time_series = []

        # Go through each of the timesteps
        for time_step in range(len(list(img_t1_all))):
            # Get the data for the particular timestep
            img_t1 = list(img_t1_all)[time_step]
            img_b0_d = list(img_b0_d_all)[time_step]
            img_b0_u = list(img_b0_u_all)[time_step]
            img_mask = list(img_mask_all)[time_step]
            img_data = np.stack((img_b0_d, img_t1))
            img_fieldmap = list(img_fieldmap_all)[time_step]
            b0u_affine = list(b0u_affine_all)[time_step]
            fieldmap_affine = list(fieldmap_affine_all)[time_step]
            echo_spacing = list(echo_spacing_all)[time_step]

            # Add the data to the time-series
            time_series.append({
                'img': img_data,
                't1': img_t1,
                'b0d': img_b0_d,
                'b0u': img_b0_u,
                'mask': img_mask,
                'fieldmap': img_fieldmap,
                'b0u_affine': b0u_affine,
                'fieldmap_affine': fieldmap_affine,
                'echo_spacing': echo_spacing
            })

        # Return the time-series
        return time_series

    # Undistorting the image
    def get_undistorted_b0(self, sample):
        # Get the data
        input_img = torch.as_tensor(sample['img']).float().to(self.device)
        input_b0d = torch.as_tensor(sample['b0d']).float().to(self.device)
        input_b0u_affine = torch.as_tensor(sample['b0u_affine']).float().to(self.device)
        input_fieldmap_affine = torch.as_tensor(sample['fieldmap_affine']).float().to(self.device)
        input_echo_spacing = torch.as_tensor(sample['echo_spacing']).float().to(self.device)

        # Estimated output fieldmap
        out = self.model(input_img.unsqueeze(0))

        # Undistorted (unwarped) output
        result = self.model._undistort_b0(input_b0d, out[0], input_b0u_affine, input_fieldmap_affine, input_echo_spacing)

        # Convert to torch and output
        return data_util.nii2torch(result)[0], out[0].detach().cpu().numpy()
