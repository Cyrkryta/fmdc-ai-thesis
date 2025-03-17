# Import of all necessary dependencies
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import volumentations
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import pad
import json
from project.data import fmri_data_util
from project.data import augmentations
import argparse
import glob

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
        self.img_b0_d_alltf = [] # Remains empty if fulltf not provided
        self.img_b0_u_alltf = [] # Remains empty if fulltf not provided

        # Go through each subject in the paths
        for SUBJECT_PATH in SUBJECT_PATHS:
            # Retrieve the tuple
            img_t1, img_b0_d, img_b0_u, img_mask, img_fieldmap, b0u_affine, b0d_affine, fieldmap_affine, echo_spacing, img_b0alltf_d, img_b0alltf_u = fmri_data_util.load_data_from_path(SUBJECT_PATH)

            # Split images on temporal access into corresponding independent samples
            if split_temporally:
                self.img_t1.extend(list(img_t1))
                self.img_b0_d.extend(list(img_b0_d))
                self.img_b0_u.extend(list(img_b0_u))
                self.img_mask.extend(list(img_mask))
                self.img_fieldmap.extend(list(img_fieldmap))
                self.b0u_affine.extend(list(b0u_affine))
                self.b0d_affine.extend(list(b0d_affine))
                self.fieldmap_affine.extend(list(fieldmap_affine))
                self.echo_spacing.extend(list(echo_spacing))

                # Append extra test files only if they exist
                if img_b0alltf_d is not None and img_b0alltf_u is not None:
                    self.img_b0_d_alltf.extend(list(img_b0alltf_d))
                    self.img_b0_u_alltf.extend(list(img_b0alltf_u))
                else:
                    pass

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
"""
Module for creating the datasets
"""
class FMRIDataModule(pl.LightningDataModule):
    # Initialize the necessary components
    def __init__(self, TRAIN_DATASET_PATHS, DATASET_SAVE_ROOT, device, TEST_DATASET_PATHS=None):
        # Setup parameters and load data immediately on initialization
        super(FMRIDataModule).__init__()
        self.TRAIN_DATASET_PATHS = TRAIN_DATASET_PATHS
        self.DATASET_SAVE_ROOT = DATASET_SAVE_ROOT
        self.TEST_DATASET_PATHS = TEST_DATASET_PATHS
        # self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
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
            # train_count = int(0.8 * total_train_count)
            train_count = int(0.1 * total_train_count)
            # val_count = total_train_count - train_count
            val_count = int(0.1 * total_train_count)
            remaining = total_train_count - train_count - val_count

            # Perform the seeded random split
            rng = torch.Generator()
            rng.manual_seed(0)
            train_split, val_split, _ = torch.utils.data.random_split(
                TRAIN_SUBJECT_PATHS, (train_count, val_count, remaining), generator=rng
            )

            # Retrieve the path indices
            if hasattr(train_split, "indices"):
                TRAIN_PATHS = [TRAIN_SUBJECT_PATHS[i] for i in train_split.indices]
                VAL_PATHS = [TRAIN_SUBJECT_PATHS[i] for i in val_split.indices]
            else:
                TRAIN_PATHS = list(train_split)
                VAL_PATHS = list(val_split)
            
            # Define the test paths
            # TEST_PATHS = TEST_SUBJECT_PATHS
            TEST_PATHS = TEST_SUBJECT_PATHS[:2]

            # Print out the paths
            print(f"Train Paths: {len(TRAIN_PATHS)}")
            print(f"Validation Paths: {len(VAL_PATHS)}")
            print(f"Test Paths (from training data): {len(TEST_PATHS)}")

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
        print(f"Saving datasets...")
        try:
            torch.save(self.train_dataset, os.path.join(DATASET_SAVE_ROOT, "train_dataset.pt"))
            torch.save(self.val_dataset, os.path.join(DATASET_SAVE_ROOT, "val_dataset.pt"))
            torch.save(self.test_dataset, os.path.join(DATASET_SAVE_ROOT, "test_dataset.pt"))
        except Exception as e:
            print(f"Couldn't save the datasets: {e}")
        print(f"Datasets saved!")

if __name__ == '__main__':
    # Set up parser and arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--TRAINING_DATASET_PATH", required=True, help="Path to the training data")
    parser.add_argument("--TEST_DATASET_PATH", default=None, help="Optional: Potentiel of seperate dataset path")
    parser.add_argument("--DATASET_SAVE_ROOT", required=True, help="Path to where the dataset should be saved")

    # parse and load the arguments
    args = parser.parse_args()
    TRAINING_DATASET_PATH = args.TRAINING_DATASET_PATH
    TEST_DATASET_PATH = args.TEST_DATASET_PATH
    DATASET_SAVE_ROOT = args.DATASET_SAVE_ROOT

    # Expand the provided dataset paths
    TRAINING_DATASET_PATHS = glob.glob(TRAINING_DATASET_PATH)

    if TEST_DATASET_PATH is not None:
        TEST_DATASET_PATHS = glob.glob(TEST_DATASET_PATH)
    else:
        TEST_DATASET_PATHS=None

    # setting the device
    device = 'cpu'
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn')
        device = 'cuda'
        print(f'Running on {device}!')

    # Create the datset module
    FMRIDataModule(
        TRAIN_DATASET_PATHS=TRAINING_DATASET_PATHS,
        DATASET_SAVE_ROOT=DATASET_SAVE_ROOT,
        device=device,
        TEST_DATASET_PATHS=TEST_DATASET_PATHS
    )

    # # Set up the dataloaders
    # def train_dataloader(self):
    #     return DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, num_workers=2, shuffle=True, collate_fn=collate_fn, persistent_workers=True)

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.BATCH_SIZE, num_workers=1, shuffle=False, collate_fn=collate_fn, persistent_workers=True)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.BATCH_SIZE, num_workers=0, shuffle=False, collate_fn=collate_fn, persistent_workers=True)
