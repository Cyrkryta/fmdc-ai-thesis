import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from project.data import fmri_data_util

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
        
        for SUBJECT_PATH in self.SUBJECT_PATHS:
            # Retrieve subject information
            project = os.path.basename(os.path.dirname(SUBJECT_PATH))

            if self.mode == "train":
                img_t1, _, _, _, _, _ = fmri_data_util.load_data_from_path_for_train(SUBJECT_PATH)
                num_samples = len(img_t1)

                # Split temporally
                for t in range(num_samples):
                    self.index_mapping.append((SUBJECT_PATH, t, project))

    def __len__(self):
        return len(self.index_mapping)
    
    def __getitem__(self, idx):
        # Retrieve the information
        SUBJECT_PATH, timepoint_idx, _ = self.index_mapping[idx]
        img_t1, img_b0_d_10, img_fieldmap, _, _, _ = fmri_data_util.load_data_from_path_for_train(SUBJECT_PATH)

        # Define the data (particular timepoint)
        data = {
            "b0_d_10": img_b0_d_10[timepoint_idx, :, :, :],
            "t1": img_t1[timepoint_idx, :, :, :],
            "fieldmap": img_fieldmap[timepoint_idx, :, :, :, :] # [H, W, D]
        }



        # # Retrieve fieldmap and echo spacing
        # fieldmap_affine_item = fieldmap_affine[timepoint_idx]
        # echo_spacing_item = echo_spacing[timepoint_idx]

        # Define the 2-channel input
        img_data = torch.stack((torch.from_numpy(data['b0_d_10']).float().to(self.device), torch.from_numpy(data['t1']).float().to(self.device)), dim=0)
        print(img_data.shape)
        fieldmap = torch.from_numpy(data["fieldmap"]).float().to(self.device)
        print(fieldmap.shape)

        output = {
            "img_data": img_data,
            "fieldmap": fieldmap,
            # "fieldmap_affine": torch.from_numpy(fieldmap_affine_item).float().to(self.device),
            # "echo_spacing": torch.from_numpy(np.asarray(echo_spacing_item)).float().to(self.device)
            # "unwarp_direction": unwarp_direction
        }

        return output


