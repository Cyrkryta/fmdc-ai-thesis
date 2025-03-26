# Importing all relevant dependencies
import os
import torch
from torch.utils.data import DataLoader, Dataset

import os
from torch.utils.data import DataLoader, Dataset, random_split, Sampler
from pytorch_lightning.trainer.supporters import CombinedLoader
from project.data import fmri_data_util
import torch
from project.data import augmentations
import numpy as np
import pytorch_lightning as pl
import json
import random



def save_to_json(TRAIN_PATHS, VAL_PATHS, TEST_PATHS):
    """
    Function for saving train, validation, and validation paths to json
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
    Class for loading and creating data
    """
    # Initialize the various components
    def __init__(self, SUBJECT_PATHS, device, augment=False, split_temporally=True):
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
        self.img_t1 = []
        self.img_b0_d = []
        self.img_b0_u = []
        self.img_mask = []
        self.img_fieldmap = []
        self.b0u_affine = []
        self.b0d_affine = []
        self.fieldmap_affine = []
        self.echo_spacing = []
        self.img_b0_d_alltf = []
        self.img_b0_u_alltf = []
        self.project_keys = []

        # Go through each subject in the paths
        for SUBJECT_PATH in SUBJECT_PATHS:
            # Extract the project key from subject
            project = os.path.basename(os.path.dirname(SUBJECT_PATH))

            # Retrieve the tuple
            img_t1, img_b0_d, img_b0_u, img_mask, img_fieldmap, b0u_affine, b0d_affine, fieldmap_affine, echo_spacing, img_b0alltf_d, img_b0alltf_u = fmri_data_util.load_data_from_path(SUBJECT_PATH)

            # Split images on temporal access into corresponding independent samples
            if split_temporally:
                num_samples = len(img_t1)
                self.img_t1.extend(list(img_t1))
                self.img_b0_d.extend(list(img_b0_d))
                self.img_b0_u.extend(list(img_b0_u))
                self.img_mask.extend(list(img_mask))
                self.img_fieldmap.extend(list(img_fieldmap))
                self.b0u_affine.extend(list(b0u_affine))
                self.b0d_affine.extend(list(b0d_affine))
                self.fieldmap_affine.extend(list(fieldmap_affine))
                self.echo_spacing.extend(list(echo_spacing))
                self.project_keys.extend([project] * num_samples)

                # Append extra test files only if they exist
                if img_b0alltf_d is not None and img_b0alltf_u is not None:
                    self.img_b0_d_alltf.extend(list(img_b0alltf_d))
                    self.img_b0_u_alltf.extend(list(img_b0alltf_u))

            # Just append the data if splitting is disabled
            else:
                self.img_t1.append(img_t1)
                self.img_b0_d.append(img_b0_d)
                self.img_b0_u.append(img_b0_u)
                self.img_mask.append(img_mask)
                self.img_fieldmap.append(img_fieldmap)
                self.b0u_affine.extend(b0u_affine)
                self.b0d_affine.extend(b0d_affine)
                self.fieldmap_affine.extend(fieldmap_affine)
                self.echo_spacing.append(echo_spacing)
                self.project_keys.append(project)

                if img_b0alltf_d is not None and img_b0alltf_u is not None:
                    self.img_b0_d_alltf.append(img_b0alltf_d)
                    self.img_b0_u_alltf.append(img_b0alltf_u)

    # Return the length of the images
    def __len__(self):
        return len(self.img_t1)

    # Get an item
    def __getitem__(self, idx):
        data = {
            'b0_d': self.img_b0_d[idx],
            'b0_u': self.img_b0_u[idx],
            't1': self.img_t1[idx],
            'mask': self.img_mask[idx],
            'fieldmap': self.img_fieldmap[idx]
        }

        # Optionally include extra test files in the data dictionary
        if self.img_b0_d_alltf and self.img_b0_u_alltf:
            data['b0alltf_d'] = self.img_b0_d_alltf[idx]
            data['b0alltf_u'] = self.img_b0_u_alltf[idx]
        
        # Perform augmentation when getting item, if retrieved
        if self.augment:
            transformed_data = self.transform(data)
        else:
            transformed_data = data

        # Stack the data to create a 2-channel input
        img_data = torch.stack((torch.from_numpy(transformed_data['b0_d']).float().to(self.device), torch.from_numpy(transformed_data['t1']).float().to(self.device)))

        # Create the output
        if 'b0alltf_d' in transformed_data and 'b0alltf_u' in transformed_data:
            output = (
                img_data,
                torch.from_numpy(transformed_data['b0_u']).float().to(self.device),
                torch.from_numpy(transformed_data['mask']).bool().to(self.device),
                torch.from_numpy(transformed_data['fieldmap']).float().to(self.device),
                torch.from_numpy(self.b0u_affine[idx]).float().to(self.device),
                torch.from_numpy(self.b0d_affine[idx]).float().to(self.device),
                torch.from_numpy(self.fieldmap_affine[idx]).float().to(self.device),
                torch.from_numpy(np.asarray(self.echo_spacing[idx])).float().to(self.device),
                torch.from_numpy(transformed_data['b0alltf_d']).float().to(self.device),
                torch.from_numpy(transformed_data['b0alltf_u']).float().to(self.device)
            )      
        else:
            output = (
                img_data,
                torch.from_numpy(transformed_data['b0_u']).float().to(self.device),
                torch.from_numpy(transformed_data['mask']).bool().to(self.device),
                torch.from_numpy(transformed_data['fieldmap']).float().to(self.device),
                torch.from_numpy(self.b0u_affine[idx]).float().to(self.device),
                torch.from_numpy(self.b0d_affine[idx]).float().to(self.device),
                torch.from_numpy(self.fieldmap_affine[idx]).float().to(self.device),
                torch.from_numpy(np.asarray(self.echo_spacing[idx])).float().to(self.device),
                None,
                None
            )

        # Return the single output
        return output
    
class FMRIDataModule(pl.LightningDataModule):
    def __init__(self, TRAIN_DATASET_PATHS, DATASET_SAVE_ROOT, device):
        super().__init__()
        self.TRAIN_DATASET_PATHS = TRAIN_DATASET_PATHS
        self.DATASET_SAVE_ROOT = DATASET_SAVE_ROOT
        self.device
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        train_path = os.path.join(self.DATASET_SAVE_ROOT, "train_dataset.pt")
        val_path = os.path.join(self.DATASET_SAVE_ROOT, "val_dataset.pt")
        
        # if the datasets exists, load them
        if os.path.exists(train_path) and os.path.exists(val_path):
            print("Loading saved datasets...")
            self.train_dataset = torch.load(train_path)
            self.val_dataset = torch.loader(val_path)
        else:
            # Otherwise create the datasets and save them
            TRAIN_SUBJECT_PATHS = fmri_data_util.collect_all_subject_paths(dataset_paths=self.TRAIN_DATASET_PATHS)
            # Splitting the data
            total_size = len(TRAIN_SUBJECT_PATHS)
            train_count = int(0.8 * total_size)
            val_count = total_size - train_count
            rng = torch.Generator()
            rng.manual_seed(0)
            train_split, val_split = random_split(TRAIN_SUBJECT_PATHS, [train_count, val_count], generator=rng)
            self.train_dataset = FMRIDataset(SUBJECT_PATHS=list(train_split), device=self.device, augment=False)
            self.val_dataset = FMRIDataset(SUBJECT_PATHS=list(val_split), device=self.device, augment=False)
            os.makedirs(self.DATASET_SAVE_ROOT, exist_ok=True)
            torch.save(self.train_dataset, train_path)
            torch.save(self.val_dataset, val_path)
            print(f"Saved train_dataset.pt ({len(self.train_dataset)} samples) and val_dataset.pt ({len(self.val_dataset)} samples)")

    def train_dataloader(self):
        # Create custom sampler
        sampler = FMRICustomSampler(self.train_dataset, batch_size=32, key_fn=get_project_key)
        # Create the loader
        train_loader = DataLoader(self.train_dataset, batch_sampler=sampler, num_workers=2)
        # Return loader
        return train_loader

    def val_dataloader(self):
        # Create custom sampler
        sampler = FMRICustomSampler(self.val_dataset, batch_size=32, key_fn=get_project_key)
        # Create the loader
        val_loader = DataLoader(self.val_dataset, batch_sampler=sampler, num_workers=1)
        # Return the loader
        return val_loader