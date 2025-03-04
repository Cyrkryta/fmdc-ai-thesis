import glob
import json

import numpy as np
import torch

from project.data import fmri_data_util, data_util
from project.metrics_scripts.metrics_computation_base import MetricsComputationBase
from project.models.unet3d_fieldmap import UNet3DFieldmap


class FieldmapsModelMetricsComputation(MetricsComputationBase):
    def __init__(self, checkpoint_path, dataset_root, device):
        super().__init__()

        self.dataset_root = dataset_root
        self.device = device

        self.model = UNet3DFieldmap.load_from_checkpoint(checkpoint_path, map_location=torch.device(device), encoder_map_location=torch.device(device), device=device)
        self.model.to(device)
        self.model.eval()

    def get_subject_paths(self):
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

    def load_input_samples(self, subject_path):
        img_t1_all, img_b0_d_all, img_b0_u_all, img_mask_all, img_fieldmap_all, b0u_affine_all, fieldmap_affine_all, echo_spacing_all = fmri_data_util.load_data_from_path(subject_path)
        time_series = []

        for time_step in range(len(list(img_t1_all))):
            img_t1 = list(img_t1_all)[time_step]
            img_b0_d = list(img_b0_d_all)[time_step]
            img_b0_u = list(img_b0_u_all)[time_step]
            img_mask = list(img_mask_all)[time_step]
            img_data = np.stack((img_b0_d, img_t1))
            img_fieldmap = list(img_fieldmap_all)[time_step]
            b0u_affine = list(b0u_affine_all)[time_step]
            fieldmap_affine = list(fieldmap_affine_all)[time_step]
            echo_spacing = list(echo_spacing_all)[time_step]

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

        return time_series

    def get_undistorted_b0(self, sample):
        input_img = torch.as_tensor(sample['img']).float().to(self.device)
        input_b0d = torch.as_tensor(sample['b0d']).float().to(self.device)
        input_b0u_affine = torch.as_tensor(sample['b0u_affine']).float().to(self.device)
        input_fieldmap_affine = torch.as_tensor(sample['fieldmap_affine']).float().to(self.device)
        input_echo_spacing = torch.as_tensor(sample['echo_spacing']).float().to(self.device)
        out = self.model(input_img.unsqueeze(0))
        result = self.model._undistort_b0(input_b0d, out[0], input_b0u_affine, input_fieldmap_affine, input_echo_spacing)
        return data_util.nii2torch(result)[0], out[0].detach().cpu().numpy()
