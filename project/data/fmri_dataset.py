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
from data import fmri_data_util
from data import data_util
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
Function for padding the tensors to the desired size
"""
def perform_padding(tensor, target_size):
    # Retrieve the current and target dimensions
    # Ignoring the channels dimension
    _, current_x, current_y, current_z = tensor.shape
    target_x, target_y, target_z = target_size

    # Calculate padding need
    padding_x = target_x - current_x
    padding_y = target_y - current_y
    padding_z = target_z - current_z

    # Calculate symmetric padding for x, y, z
    x_padding_left = padding_x // 2
    x_padding_right = padding_x - x_padding_left
    y_padding_left = padding_y // 2
    y_padding_right = padding_y - y_padding_left
    z_padding_left = padding_z // 2
    z_padding_right = padding_z - z_padding_left

    # Create the padding tuple (reversed order)
    padding_tuple = (
        x_padding_left, x_padding_right,
        y_padding_left, y_padding_right,
        z_padding_left, z_padding_right
    )

    # padding_tuple = (
    #     z_padding_left, z_padding_right,
    #     y_padding_left, y_padding_right,
    #     x_padding_left, x_padding_right
    # )

    # Perform and return the padded tensor
    # Padding value -1 as this is our normalized background
    padded_tensor = pad(input=tensor, pad=padding_tuple, mode="constant", value=-100)
    return padded_tensor

"""
Custom collate function for padding the batches (input and output)
Input shape: []
"""
def collate_fn(batch):
    # Calculate max of individual dimensions
    # Assuming target shape is the same as input shape
    max_x = max(sample[0][0].shape[0] for sample in batch)
    max_y = max(sample[0][0].shape[1] for sample in batch)
    max_z = max(sample[0][0].shape[2] for sample in batch)

    print(f"Max X Dimension: {max_x}")
    print(f"Max Y Dimension: {max_y}")
    print(f"Max Z Dimension: {max_z}")

    target_size = (max_x, max_y, max_z)

    # Placeholder for the padded batch
    padded_batch = []

    # Go through each sample
    for sample in batch:
        # Unpacking the sample
        img_data, b0_u, mask, fieldmap, b0u_affine, b0d_affine, fieldmap_affine, echo_spacing, b0alltf_d, b0alltf_u = sample

        # Retrieve the distorted EPI image and fieldmap image
        b0u_img = b0_u[0]
        fieldmap_img = fieldmap[0]

        # Check if the fieldmap and distorted epi image has similar dimensions
        if fieldmap_img.shape != b0u_img.shape:
            # If the shapes doesn't fit, resample fieldmap to epi space
            # Retrieve the nifti images
            b0u_nifti = data_util.get_nifti_image(b0u_img, b0u_affine)
            fieldmap_nifti = data_util.get_nifti_image(fieldmap_img, fieldmap_affine)

            # Resample the fieldmap image
            fieldmap_nifti_resampled = data_util.resample_image(fieldmap_nifti, b0u_nifti.affine, b0u_nifti.shape, "linear")

            # Get Resampled image data again
            fieldmap_nifti_resampled_data = fieldmap_nifti_resampled.get_fdata()
            fieldmap_resampled_img = torch.tensor(fieldmap_nifti_resampled_data, dtype=torch.float32).unsqueeze(0)
        else:
            fieldmap_resampled_img = fieldmap

            # print(f"\nb0d_img shape: {b0d_img.shape}")
            # print(f"fieldmap img shape: {fieldmap_img.shape}")
            # print(f"fieldmap resampled img shape: {fieldmap_resampled_img.shape}")

        # Pad the data
        padded_img_data = perform_padding(img_data, target_size)
        padded_b0_u = perform_padding(b0_u, target_size)
        padded_mask = perform_padding(mask, target_size)
        padded_fieldmap = perform_padding(fieldmap_resampled_img, target_size)
        # padded_img_data = img_data
        # padded_b0_u = b0_u
        # padded_mask = mask
        # padded_fieldmap = fieldmap_resampled_img

        # print(f"\nimg_data shape: {padded_img_data.shape}")
        # print(f"b0_u shape: {padded_b0_u.shape}")
        # print(f"mask shape: {padded_mask.shape}")
        # print(f"fieldmap shape: {padded_fieldmap.shape}")

        # Create and append the padded sample
        # new_padded_sample = (padded_img_data, b0_u, mask, padded_fieldmap, affine, echo_spacing, b0alltf_d, b0alltf_u)
        new_padded_sample = (padded_img_data, padded_b0_u, padded_mask, padded_fieldmap, b0u_affine, b0d_affine, fieldmap_affine, echo_spacing, b0alltf_d, b0alltf_u)
        padded_batch.append(new_padded_sample)

    # Define a stacked dictionary
    keys = ["img_data", "b0_u", "mask", "fieldmap", "affine", "echo_spacing", "b0alltf_d", "b0alltf_u"]
    # Collated placeholder
    collated = {}
    # Handle None cases
    collated = {
        key: torch.stack(
            [sample[i] if sample [i] is not None else torch.tensor([-1], dtype=torch.float32) for sample in padded_batch]
        ) for i, key in enumerate(keys)
    }

    # # Return the new collate
    return collated

    # # unpack
    # padded_inputs = []
    # padded_targets = []

    # # Pad inputs and outputs in the batch
    # for sample in batch:
    #     # Create padded tensors
    #     padded_input = perform_padding(input, target_size)
    #     padded_target = perform_padding(target, target_size)
    #     # Append padded tensors
    #     padded_inputs.append(padded_input)
    #     padded_targets.append(padded_target)

    # # Stack and return the inputs and targets
    # stacked_padded_inputs, stacked_padded_targets = torch.stack(padded_inputs), torch.stack(padded_targets)
    # return stacked_padded_inputs, stacked_padded_targets

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

    # Set up the dataloaders
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, num_workers=2, shuffle=True, collate_fn=collate_fn, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.BATCH_SIZE, num_workers=1, shuffle=False, collate_fn=collate_fn, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.BATCH_SIZE, num_workers=0, shuffle=False, collate_fn=collate_fn, persistent_workers=True)
