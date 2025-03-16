# Import of all necessary dependencies
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import volumentations
from torch.utils.data import DataLoader, Dataset
import json
from data import fmri_data_util
from data import augmentations

"""
Function for saving train, validation, and validation paths to json
"""
def save_to_json(TRAIN_PATHS, VAL_PATHS, TEST_PATHS):
    with open("train_paths.json", "w") as f:
        json.dump({"train_paths": TRAIN_PATHS}, f, indent=4)
    with open("val_paths.json", "w") as f:
        json.dump({"val_paths": VAL_PATHS}, f, indent=4)
    with open("test_paths.json", "w") as f:
        json.dump({"test_paths": TEST_PATHS}, f, indent=4)
    print("JSON files for train, validation and test has been saved...")

"""
Class for creating the data as intended
"""
class FMRIDataset(Dataset):
    # Initialize the various components
    def __init__(self, SUBJECT_PATHS, device, augment=False, split_temporally=True):
        # Set up variables
        self.transform = augmentations.Compose([
            augmentations.Rotate((-15, 15), (-15, 15), (-15, 15)),
            augmentations.ElasticTransform((0, 0.25), interpolation=2),
            augmentations.RandomScale([0.9, 1.1], interpolation=1),
            augmentations.Resize((36, 64, 64), interpolation=1, resize_type=0)
        ])
        self.device = device
        self.augment = augment
        self.split_temporally = split_temporally
        self.SUBJECT_PATHS = SUBJECT_PATHS
        self.index_mapping = []

        # Go through each subject
        for SUBJECT_PATH in self.SUBJECT_PATHS:
            # Retrieve a small amount of data
            img_t1, _, _, _, _, _, _, _, _, _ = fmri_data_util.load_data_from_path(SUBJECT_PATH)

            # Retrieve a time_seq
            try:
                n_timepoints = len(img_t1)
            except Exception:
                n_timepoints = 1
            for t in range(n_timepoints):
                self.index_mapping.append((SUBJECT_PATH, t))

    # Return the length of the images
    def __len__(self):
        return len(self.index_mapping)

    # Get an item
    def __getitem__(self, idx):
        # Get the subject path from the mapping
        SUBJECT_PATH, t_idx = self.index_mapping[idx]

        # Load the fulld subject data (on demand)
        img_t1, img_b0_d, img_b0_u, img_mask, img_fieldmap, b0u_affine, _, echo_spacing, img_b0alltf_d, img_b0alltf_u = fmri_data_util.load_data_from_path(SUBJECT_PATH)

        # Select the timeslice
        data = {
            't1': img_t1[t_idx],
            'b0_d': img_b0_d[t_idx],
            'b0_u': img_b0_u[t_idx],
            'mask': img_mask[t_idx],
            'fieldmap': img_fieldmap[t_idx]
        }

        # Optionally include the extra files if they exist.
        if (img_b0alltf_d is not None) and (img_b0alltf_u is not None):
            data['b0alltf_d'] = img_b0alltf_d[t_idx]
            data['b0alltf_u'] = img_b0alltf_u[t_idx]

        # Grab affine and echo spacing for this timepoint.
        affine = b0u_affine[t_idx]
        echo_sp = echo_spacing[t_idx]
        
        # Apply augmentation if enabled.
        if self.augment:
            transformed_data = self.transform(data)
        else:
            transformed_data = data

        # Stack to create a 2-channel input: distorted and T1 images.
        img_data = torch.stack((
            torch.from_numpy(transformed_data['b0_d']).float().to(self.device),
            torch.from_numpy(transformed_data['t1']).float().to(self.device)
        ))

        output = (
            img_data,
            torch.from_numpy(transformed_data['b0_u']).float().to(self.device),
            torch.from_numpy(transformed_data['mask']).bool().to(self.device),
            torch.from_numpy(transformed_data['fieldmap']).float().to(self.device),
            torch.from_numpy(affine).float().to(self.device),
            torch.from_numpy(np.asarray(echo_sp)).float().to(self.device)
        )

        if ('b0alltf_d' in transformed_data) and ('b0alltf_u' in transformed_data):
            extra = (
                torch.from_numpy(transformed_data['b0alltf_d']).float().to(self.device),
                torch.from_numpy(transformed_data['b0alltf_u']).float().to(self.device)
            )
            return output + extra

        return output
    
"""
Module for creating the datasets
"""
class FMRIDataModule(pl.LightningDataModule):
    # Initialize the necessary components
    def __init__(self, TRAIN_DATASET_PATHS, BATCH_SIZE, device, TEST_DATASET_PATHS=None):
        # Setup parameters and load data immediately on initialization
        super(FMRIDataModule).__init__()
        self.TRAIN_DATASET_PATHS = TRAIN_DATASET_PATHS
        self.TEST_DATASET_PATHS = TEST_DATASET_PATHS
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.metrics_dataset = None
        self.load_data()

    # Function for loading the data
    def load_data(self):
        # If test dataset doesn't exist, split training data only
        if self.TEST_DATASET_PATHS is None:
            # Retrieve paths
            TRAIN_SUBJECT_PATHS = fmri_data_util.collect_all_subject_paths(dataset_paths=self.TRAIN_DATASET_PATHS)

            # Split the data (70/20/10)
            total_train_count = len(TRAIN_SUBJECT_PATHS)
            train_count = int(0.7 * total_train_count)
            val_count = int(0.2 * total_train_count)
            test_count = total_train_count - train_count - val_count

            # Perform the seeded random split
            rng = torch.Generator()
            rng.manual_seed(0)
            train_split, val_split, test_split = torch.utils.data.random_split(
                TRAIN_SUBJECT_PATHS, (train_count, val_count, test_count), generator=rng
            )

            # Retrieve the path indices
            if hasattr(train_split, "indices"):
                TRAIN_PATHS = [TRAIN_SUBJECT_PATHS[i] for i in train_split.indices]
                VAL_PATHS = [TRAIN_SUBJECT_PATHS[i] for i in val_split.indices]
                TEST_PATHS = [TRAIN_SUBJECT_PATHS[i] for i in test_split.indices]
            else:
                TRAIN_PATHS = list(train_split)
                VAL_PATHS = list(val_split)
                TEST_PATHS = list(test_split)
            
            # Print out the paths
            print(f"Train Paths: {len(TRAIN_PATHS)}")
            print(f"Validation Paths: {len(VAL_PATHS)}")
            print(f"Test Paths (from training data): {len(TEST_PATHS)}")
        # If test dataset exists
        else:
            # Retrieve paths
            print(self.TRAIN_DATASET_PATHS)
            TRAIN_SUBJECT_PATHS = fmri_data_util.collect_all_subject_paths(dataset_paths=self.TRAIN_DATASET_PATHS)
            TEST_SUBJECT_PATHS = fmri_data_util.collect_all_subject_paths(dataset_paths=self.TEST_DATASET_PATHS)

            # Split the data
            total_train_count = len(TRAIN_SUBJECT_PATHS)
            train_count = int(0.8 * total_train_count)
            # train_count = int(0.8 * total_train_count)
            val_count = total_train_count - train_count
            # val_count = int(0.2 * total_train_count)
            # remaining = total_train_count - train_count - val_count

            # Perform the seeded random split
            rng = torch.Generator()
            rng.manual_seed(0)
            train_split, val_split = torch.utils.data.random_split(
                TRAIN_SUBJECT_PATHS, (train_count, val_count), generator=rng
            )

            # Retrieve the path indices
            if hasattr(train_split, "indices"):
                TRAIN_PATHS = [TRAIN_SUBJECT_PATHS[i] for i in train_split.indices]
                VAL_PATHS = [TRAIN_SUBJECT_PATHS[i] for i in val_split.indices]
            else:
                TRAIN_PATHS = list(train_split)
                VAL_PATHS = list(val_split)
            
            # Define the test paths
            TEST_PATHS = TEST_SUBJECT_PATHS[:2]
            # TEST_PATHS = TEST_SUBJECT_PATHS[:2]

            # Print out the paths
            print(f"Train Paths: {len(TRAIN_PATHS)}")
            print(f"Validation Paths: {len(VAL_PATHS)}")
            print(f"Test Paths (from training data): {len(TEST_PATHS)}")

        # Save test paths as an attribute
        self.TEST_PATHS = TEST_PATHS

        # Save the splits to json
        save_to_json(TRAIN_PATHS=TRAIN_PATHS, VAL_PATHS=VAL_PATHS, TEST_PATHS=TEST_PATHS)

        # Create datasets from the splits
        print("Before training dataset")
        self.train_dataset = FMRIDataset(SUBJECT_PATHS=TRAIN_PATHS, device=self.device, augment=False)
        print("Before validation dataset")
        self.val_dataset = FMRIDataset(SUBJECT_PATHS=VAL_PATHS, device=self.device, augment=False)
        print("Before test dataset")
        self.test_dataset = FMRIDataset(SUBJECT_PATHS=TEST_PATHS, device=self.device, augment=False)

        # Output the current splits for the datasets
        print(f'Loaded datasets: {len(self.train_dataset)} training samples, {len(self.val_dataset)} validation samples, {len(self.test_dataset)} test samples.')

    # Set up the dataloaders
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, num_workers=2, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.BATCH_SIZE, num_workers=1, shuffle=False, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.BATCH_SIZE, num_workers=0, shuffle=False, persistent_workers=True)
