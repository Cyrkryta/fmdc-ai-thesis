import numpy as np
import torch

from project.data import fmri_data_util, data_util
from project.metrics_scripts.metrics_computation_base import MetricsComputationBase
from project.models.unet3d_fieldmap import UNet3DFieldmap

class KFoldFieldmapsModelMetricsComputation(MetricsComputationBase):
    def __init__(self, CHECKPOINT_PATH, TEST_PATHS, device):
        super().__init__()

        self.TEST_PATHS = TEST_PATHS
        self.device = device
        self.model = UNet3DFieldmap.load_from_checkpoint(CHECKPOINT_PATH, map_location=torch.device(device), encoder_map_location=torch.device(device), device=device)
        self.model.to(device)
        self.model.eval()

    def get_subject_paths(self):
        return self.TEST_PATHS

    def load_input_samples(self, subject_path):
        t1w, b0d, b0u, b0_mask, fieldmap, b0d_affine, fieldmap_affine, echospacing, phaseencodingdirection = fmri_data_util.load_data_from_path_for_test(subject_path=subject_path)

        # placeholder
        time_series = []

        # Go through each of the timesteps data
        for t_step in range(len(list(t1w))):
            img_t1w = list(t1w)[t_step]
            img_b0d = list(b0d)[t_step]
            img_b0u = list(b0u)[t_step]
            img_b0mask = list(b0_mask)[t_step]
            img_data = np.stack((img_b0d, img_t1w))
            img_fieldmap = list(fieldmap)[t_step]
            b0d_affine_list = list(b0d_affine)[t_step]
            fieldmap_affine_list = list(fieldmap_affine)[t_step]
            echospacing_val = echospacing
            phaseencodingdirection_val = phaseencodingdirection

            # Append to the time_series
            time_series.append({
                'img': img_data,
                't1': img_t1w,
                'b0d': img_b0d,
                'b0u': img_b0u,
                'mask': img_b0mask,
                'fieldmap': img_fieldmap,
                'b0d_affine': b0d_affine_list,
                "fieldmap_affine": fieldmap_affine_list,
                'echo_spacing': echospacing_val,
                'phase_encoding_direction': phaseencodingdirection_val
            })

        # img_t1_all, img_b0_d_all, img_b0_u_all, img_mask_all, img_fieldmap_all, b0u_affine_all, fieldmap_affine_all, echo_spacing_all = fmri_data_util.load_data_from_path(subject_path)
        # time_series = []

        # for time_step in range(len(list(img_t1_all))):
        #     img_t1 = list(img_t1_all)[time_step]
        #     img_b0_d = list(img_b0_d_all)[time_step]
        #     img_b0_u = list(img_b0_u_all)[time_step]
        #     img_mask = list(img_mask_all)[time_step]
        #     img_data = np.stack((img_b0_d, img_t1))
        #     img_fieldmap = list(img_fieldmap_all)[time_step]
        #     b0u_affine = list(b0u_affine_all)[time_step]
        #     fieldmap_affine = list(fieldmap_affine_all)[time_step]
        #     echo_spacing = list(echo_spacing_all)[time_step]

            # time_series.append({
            #     'img': img_data,
            #     't1': img_t1,
            #     'b0d': img_b0_d,
            #     'b0u': img_b0_u,
            #     'mask': img_mask,
            #     'fieldmap': img_fieldmap,
            #     'b0u_affine': b0u_affine,
            #     'fieldmap_affine': fieldmap_affine,
            #     'echo_spacing': echo_spacing
            # })

        return time_series

def get_undistorted_b0(self, sample):
        # Get the data
        input_img = torch.as_tensor(sample['img']).float().to(self.device)
        input_b0d = torch.as_tensor(sample['b0d']).float().to(self.device)
        input_b0d_affine = torch.as_tensor(sample['b0d_affine']).float().to(self.device)
        # input_b0u_affine = torch.as_tensor(sample['b0u_affine']).float().to(self.device)
        input_fieldmap_affine = torch.as_tensor(sample['fieldmap_affine']).float().to(self.device)
        input_echo_spacing = sample["echo_spacing"]
        input_unwarp_direction = sample["phase_encoding_direction"]
        # input_echo_spacing = torch.as_tensor(sample['echo_spacing']).float().to(self.device)
        # input_unwarp_direction = torch.as_tensor(sample['phase_encoding_direction']).float().to(self.device)

        # Estimated output fieldmap
        out = self.model(input_img.unsqueeze(0))

        # Undistorted (unwarped) output
        result_u = self.model._undistort_b0(
            input_b0d,
            out[0],
            input_b0d_affine,
            input_fieldmap_affine,
            input_echo_spacing,
            input_unwarp_direction
        )
        # result = self.model._undistort_b0(input_b0d, out[0], input_b0u_affine, input_fieldmap_affine, input_echo_spacing)

        # Convert to torch and output
        return data_util.nii2torch(result_u)[0], out[0].detach().cpu().numpy()

    # def get_undistorted_b0(self, sample):
    #     input_img = torch.as_tensor(sample['img']).float().to(self.device)
    #     input_b0d = torch.as_tensor(sample['b0d']).float().to(self.device)
    #     input_b0u_affine = torch.as_tensor(sample['b0u_affine']).float().to(self.device)
    #     input_fieldmap_affine = torch.as_tensor(sample['fieldmap_affine']).float().to(self.device)
    #     input_echo_spacing = torch.as_tensor(sample['echo_spacing']).float().to(self.device)
    #     out = self.model(input_img.unsqueeze(0))
    #     result = self.model._undistort_b0(input_b0d, out[0], input_b0u_affine, input_fieldmap_affine, input_echo_spacing)
    #     return data_util.nii2torch(result)[0], out[0].detach().cpu().numpy()
