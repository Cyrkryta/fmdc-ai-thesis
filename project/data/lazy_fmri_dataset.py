import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from project.data import fmri_data_util
import nibabel as nib

def save_to_json(TRAIN_PATHS, VAL_PATHS, TEST_PATHS):
    """
    Save the subject paths to a json file
    """
    with open("train_paths.json", "w") as f:
        json.dump({"train_paths": TRAIN_PATHS}, f, indent=4)
    with open("val_paths.json", "w") as f:
        json.dump({"val_paths": VAL_PATHS}, f, indent=4)
    with open("test_paths.json", "w") as f:
        json.dump({"test_paths": TEST_PATHS}, f, indent=4)
    print("JSON files for train, validation and test has been saved...")

class LazyFMRIDataset(Dataset):
    """
    Dataset for performing lazy loading
    """
    def __init__(self, SUBJECT_PATHS, device, mode="train"):
        self.SUBJECT_PATHS = SUBJECT_PATHS
        self.device = device
        self.mode = mode
        self.index_mapping = []

        if self.mode == "train":
            for SUBJECT_PATH in self.SUBJECT_PATHS:
                # Retrieve subject information
                project = os.path.basename(os.path.dirname(SUBJECT_PATH))

                # img_t1, _, _, _, _, _ = fmri_data_util.load_data_from_path_for_train(SUBJECT_PATH)
                img_t1, _, _, _ = fmri_data_util.load_data_from_path_for_train(SUBJECT_PATH, use_saved_nifti=True)
                num_samples = len(img_t1)

                # Split temporally
                for t in range(num_samples):
                    self.index_mapping.append((SUBJECT_PATH, t, project))

    def __len__(self):
        return len(self.index_mapping)
    
    def __getitem__(self, idx):
        # Retrieve the information
        SUBJECT_PATH, timepoint_idx, _ = self.index_mapping[idx]
        if self.mode == "train":
            # img_t1, img_b0_d_10, img_fieldmap, _, _, _ = fmri_data_util.load_data_from_path_for_train(SUBJECT_PATH)
            # img_t1, img_b0_d_10, img_fieldmap = fmri_data_util.load_data_from_path_for_train(SUBJECT_PATH, use_saved_nifti=True)
            img_t1, img_b0_d_10, img_fieldmap, img_mask = fmri_data_util.load_data_from_path_for_train(SUBJECT_PATH, use_saved_nifti=True)
            threshold = -0.9
            mask = (img_t1[timepoint_idx] > threshold).astype(np.float32)
            # print(img_mask.shape)
            # t1w = os.path.join(SUBJECT_PATH, "T1w.nii.gz")
            # t1w = nib.load(t1w).get_fdata()
            # t1w = np.transpose(t1w, axes=(2, 0, 1))
            # print(f"Min value in t1w orig: {np.min(t1w)}, Max value in t1w orig: {np.max(t1w)}")
            # print(f"Min value in t1w proces: {np.min(img_t1)}, Max value in t1w proces: {np.max(img_t1)}")
            # threshold = -0.9
            # t1_mask = img_t1[timepoint_idx, :, :, :]
            # t1_mask = t1_mask > threshold
            
            # t1_mask =  > threshold
            # print(f"{img_t1.shape}")
            # t1_img = img_t1[timepoint_idx, :, :, :]
            # threshold = 0.05
            # t1_mask = t1_img > threshold
            # print(f"\n{img_t1.shape}")
            # print(f"{t1_mask.shape}")
            # print(f"{img_mask.shape}")
            # t1_mask = t1_mask.unsqueeze(0)

            # Define the data (particular timepoint)
            data = {
                "b0_d_10": img_b0_d_10[timepoint_idx, :, :, :],
                "t1": img_t1[timepoint_idx, :, :, :],
                "fieldmap": img_fieldmap[timepoint_idx, :, :, :, :], # [H, W, D]
                # "mask": img_mask[timepoint_idx, :, :, :]
                "mask": mask
            }

            # Define the 2-channel input
            # img_data = torch.stack((torch.from_numpy(data['b0_d_10']).float().to(self.device), torch.from_numpy(data['t1']).float().to(self.device)))
            img_data = torch.stack((torch.from_numpy(data['b0_d_10']).float(), torch.from_numpy(data['t1']).float()))
            # print(img_data.shape)
            # print(np.min(data["fieldmap"]), np.max(data["fieldmap"]))
            # fieldmap = torch.from_numpy(data["fieldmap"]).float().to(self.device)
            fieldmap = torch.from_numpy(data["fieldmap"]).float()
            # print(fieldmap.shape)
            # mask = torch.from_numpy(data["mask"]).float().to(self.device)
            mask = torch.from_numpy(data["mask"]).unsqueeze(0).float()
            # mask = mask.unsqueeze(0)
            # mask = mask.unsqueeze(0)
            # print(mask.shape)

            output = {
                "img_data": img_data,
                "fieldmap": fieldmap,
                "mask": mask
            }

            return output

            # # Retrieve fieldmap and echo spacing
            # fieldmap_affine_item = fieldmap_affine[timepoint_idx]
            # echo_spacing_item = echo_spacing[timepoint_idx]
            # "fieldmap_affine": torch.from_numpy(fieldmap_affine_item).float().to(self.device),
            # "echo_spacing": torch.from_numpy(np.asarray(echo_spacing_item)).float().to(self.device)
            # "unwarp_direction": unwarp_direction
