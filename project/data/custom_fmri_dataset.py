import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from project.data import fmri_data_util, augmentations

def save_to_json(TRAIN_PATHS, VAL_PATHS, TEST_PATHS):
    """
    Function for saving paths to json
    """
    with open("train_paths.json", "w") as f:
        json.dump({"train_paths": TRAIN_PATHS}, f, indent=4)
    with open("val_paths.json", "w") as f:
        json.dump({"val_paths": VAL_PATHS}, f, indent=4)
    with open("test_paths.json", "w") as f:
        json.dump({"test_paths": TEST_PATHS}, f, indent=4)
    print("JSON files for train, validation and test has been saved...")


class FMRIDataset(Dataset):
    """
    Class for creating the data as intended
    """
    # Initialize the various components
    def __init__(self, SUBJECT_PATHS, device, augment=False, split_temporally=True, mode="train"):
        # Transformation component for augmentations
        self.transform = augmentations.Compose([
            augmentations.Rotate((-15, 15), (-15, 15), (-15, 15)),
            augmentations.ElasticTransform((0, 0.25), interpolation=2),
            augmentations.RandomScale([0.9, 1.1], interpolation=1),
            augmentations.Resize((36, 64, 64), interpolation=1, resize_type=0)
        ])
        # Set device and augmentation state
        self.device = device
        self.augment = augment
        # Placeholders for the data output
        if mode=="train":
            self.img_t1 = []
            self.img_b0_d_10 = []
            self.img_fieldmap = []
            self.fieldmap_affine = []
            self.echo_spacing = []
            self.unwarp_direction = []
            self.project_keys = []

        # Go through each subject in the paths
        for SUBJECT_PATH in SUBJECT_PATHS:
            # Retrieve info
            project = os.path.basename(os.path.dirname(SUBJECT_PATH))

            if mode == "train":
                img_t1, img_b0_d_10, img_fieldmap, fieldmap_affine, echo_spacing, unwarp_direction = fmri_data_util.load_data_from_path_for_train(SUBJECT_PATH)
                num_samples = len(img_t1)
                if split_temporally:
                    self.img_t1.extend(list(img_t1))
                    self.img_b0_d_10.extend(list(img_b0_d_10))
                    self.img_fieldmap.extend(list(img_fieldmap))
                    self.fieldmap_affine.extend(list(fieldmap_affine))
                    self.echo_spacing.extend(list(echo_spacing))
                    self.unwarp_direction.extend(list(unwarp_direction))
                    self.project_keys.extend([project] * num_samples)

    # Return the length of the images
    def __len__(self):
        return len(self.img_t1)

    # Get an item
    def __getitem__(self, idx):
        data = {
            'b0_d_10': self.img_b0_d_10[idx],
            # 'b0_u': self.img_b0_u[idx],
            't1': self.img_t1[idx],
            # 'mask': self.img_mask[idx],
            'fieldmap': self.img_fieldmap[idx]
        }
        
        # Perform augmentation when getting item, if retrieved
        if self.augment:
            transformed_data = self.transform(data)
        else:
            transformed_data = data

        # Stack the data to create a 2-channel input
        img_data = torch.stack((torch.from_numpy(transformed_data['b0_d_10']).float().to(self.device), torch.from_numpy(transformed_data['t1']).float().to(self.device)))

        output = {
            "img_data": img_data,
            "fieldmap": torch.from_numpy(transformed_data["fieldmap"]).float().to(self.device),
            "fieldmap_affine": torch.from_numpy(self.fieldmap_affine[idx]).float().to(self.device),
            "echo_spacing": torch.from_numpy(np.asarray(self.echo_spacing[idx])).float().to(self.device),
            "unwarp_direction": self.unwarp_direction
        }


        # Return the single output
        return output