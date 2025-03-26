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
        # else:
        #     self.img_b0_d = []
        #     self.img_b0_u = []
        #     self.img_mask = []
        #     self.img_fieldmap = []
        #     self.b0u_affine = []
        #     self.b0d_affine = []
        #     self.fieldmap_affine = []
        #     self.echo_spacing = []
        #     self.img_b0_d_alltf = []
        #     self.img_b0_u_alltf = []
        #     self.project_keys = []

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
            # else:
            #     tuple = fmri_data_util.load_data_from_path(SUBJECT_PATH)
            # img_t1, img_b0_d, img_b0_u, img_mask, img_fieldmap, b0u_affine, b0d_affine, fieldmap_affine, echo_spacing, img_b0alltf_d, img_b0alltf_u = fmri_data_util.load_data_from_path(SUBJECT_PATH)

            # # Split images on temporal access into corresponding independent samples
            # if split_temporally:
            #     self.img_t1.extend(list(img_t1))
            #     self.img_b0_d.extend(list(img_b0_d))
            #     self.img_b0_u.extend(list(img_b0_u))
            #     self.img_mask.extend(list(img_mask))
            #     self.img_fieldmap.extend(list(img_fieldmap))
            #     self.b0u_affine.extend(list(b0u_affine))
            #     self.b0d_affine.extend(list(b0d_affine))
            #     self.fieldmap_affine.extend(list(fieldmap_affine))
            #     self.echo_spacing.extend(list(echo_spacing))
            #     self.project_keys.extend([project])
            #     # Append extra test files only if they exist
            #     if img_b0alltf_d is not None and img_b0alltf_u is not None:
            #         self.img_b0_d_alltf.extend(list(img_b0alltf_d))
            #         self.img_b0_u_alltf.extend(list(img_b0alltf_u))
            #     else:
            #         pass

            # Just append the data if splitting is disabled
            # else:
            #     self.img_t1.append(img_t1)
            #     self.img_b0_d.append(img_b0_d)
            #     self.img_b0_u.append(img_b0_u)
            #     self.img_mask.append(img_mask)
            #     self.img_fieldmap.append(img_fieldmap)
            #     self.b0u_affine.extend(b0u_affine)
            #     self.b0d_affine.extend(b0d_affine)
            #     self.fieldmap_affine.extend(fieldmap_affine)
            #     self.echo_spacing.append(echo_spacing)
            #     self.project_keys.append(project)
            #     if img_b0alltf_d is not None and img_b0alltf_u is not None:
            #         self.img_b0_d_alltf.append(img_b0alltf_d)
            #         self.img_b0_u_alltf.append(img_b0alltf_u)

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

        # Optionally include extra test files in the data dictionary
        # if self.img_b0_d_alltf and self.img_b0_u_alltf:
        #     data['b0alltf_d'] = self.img_b0_d_alltf[idx]
        #     data['b0alltf_u'] = self.img_b0_u_alltf[idx]
        
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
            # img_data,
            # torch.from_numpy(transformed_data["fieldmap"]).float().to(self.device),
            # torch.from_numpy(self.fieldmap_affine[idx]).float().to(self.device),
            # torch.from_numpy(np.asarray(self.echo_spacing[idx])).float().to(self.device)
        
        # # Create the output
        # if 'b0alltf_d' in transformed_data and 'b0alltf_u' in transformed_data:
        #     output = (
        #         img_data,
        #         torch.from_numpy(transformed_data['b0_u']).float().to(self.device),
        #         torch.from_numpy(transformed_data['mask']).bool().to(self.device),
        #         torch.from_numpy(transformed_data['fieldmap']).float().to(self.device),
        #         torch.from_numpy(self.b0u_affine[idx]).float().to(self.device),
        #         torch.from_numpy(self.b0d_affine[idx]).float().to(self.device),
        #         torch.from_numpy(self.fieldmap_affine[idx]).float().to(self.device),
        #         torch.from_numpy(np.asarray(self.echo_spacing[idx])).float().to(self.device),
        #         torch.from_numpy(transformed_data['b0alltf_d']).float().to(self.device),
        #         torch.from_numpy(transformed_data['b0alltf_u']).float().to(self.device)
        #     )      
        # else:
        #     output = (
        #         img_data,
        #         torch.from_numpy(transformed_data['b0_u']).float().to(self.device),
        #         torch.from_numpy(transformed_data['mask']).bool().to(self.device),
        #         torch.from_numpy(transformed_data['fieldmap']).float().to(self.device),
        #         torch.from_numpy(self.b0u_affine[idx]).float().to(self.device),
        #         torch.from_numpy(self.b0d_affine[idx]).float().to(self.device),
        #         torch.from_numpy(self.fieldmap_affine[idx]).float().to(self.device),
        #         torch.from_numpy(np.asarray(self.echo_spacing[idx])).float().to(self.device),
        #         None,
        #         None
        #     )

        # Return the single output
        return output