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
            img_t1, img_b0_d_10, img_fieldmap, img_mask = fmri_data_util.load_data_from_path_for_train(SUBJECT_PATH, use_saved_nifti=True)
            threshold = -0.9
            mask = (img_t1[timepoint_idx] > threshold).astype(np.float32)
            data = {
                "b0_d_10": img_b0_d_10[timepoint_idx, :, :, :],
                "t1": img_t1[timepoint_idx, :, :, :],
                "fieldmap": img_fieldmap[timepoint_idx, :, :, :, :],
                "mask": mask
            }
            img_data = torch.stack((torch.from_numpy(data['b0_d_10']).float(), torch.from_numpy(data['t1']).float()))
            fieldmap = torch.from_numpy(data["fieldmap"]).float()
            mask = torch.from_numpy(data["mask"]).unsqueeze(0).float()

            output = {
                "img_data": img_data,
                "fieldmap": fieldmap,
                "mask": mask
            }

            return output