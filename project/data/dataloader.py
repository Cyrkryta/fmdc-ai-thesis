# importing the necessary dependencies
import torch
from project.data import data_util
from project.data.fmri_dataset import FMRIDataset
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from os import path

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

    padding_tuple = (
        z_padding_left, z_padding_right,
        y_padding_left, y_padding_right,
        x_padding_left, x_padding_right
    )

    # Perform and return the padded tensor
    # Padding value -1 as this is our normalized background
    padded_tensor = pad(input=tensor, pad=padding_tuple, mode="constant", value=-100)
    return padded_tensor

"""
Round up to a suitable multiple
"""
def round_up_to_multiple(x, factor):
    new_x = ((x + factor -1) // factor) * factor
    return new_x

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

    # max_x, max_y, max_z = 100, 100, 100

    # Defining the target size
    target_size = (max_x, max_y, max_z)
    print(f"Padded target size: {target_size}")

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
            fieldmap_nifti_resampled = data_util.resample_image(fieldmap_nifti, b0u_nifti.affine, b0u_nifti.shape)

            # Get Resampled image data again
            fieldmap_nifti_resampled_data = fieldmap_nifti_resampled.get_fdata()
            fieldmap_resampled_img = torch.tensor(fieldmap_nifti_resampled_data, dtype=torch.float32).unsqueeze(0)
        else:
            fieldmap_resampled_img = fieldmap

        # Pad the data
        padded_img_data = perform_padding(img_data, target_size)
        padded_b0_u = perform_padding(b0_u, target_size)
        padded_mask = perform_padding(mask, target_size)
        padded_fieldmap = perform_padding(fieldmap_resampled_img, target_size)

        # Create and append the padded sample
        # new_padded_sample = (padded_img_data, b0_u, mask, padded_fieldmap, affine, echo_spacing, b0alltf_d, b0alltf_u)
        new_padded_sample = (padded_img_data, padded_b0_u, padded_mask, padded_fieldmap, b0u_affine, b0d_affine, fieldmap_affine, echo_spacing, b0alltf_d, b0alltf_u)
        padded_batch.append(new_padded_sample)

    # Define a stacked dictionary
    keys = ["img_data", "b0_u", "mask", "fieldmap", "b0u_affine", "b0d_affine", "fieldmap_affine", "echo_spacing", "b0alltf_d", "b0alltf_u"]
    # Collated placeholder
    collated = {}
    # Handle None cases
    collated = {
        key: torch.stack(
            [sample[i] if sample [i] is not None else torch.tensor([-1], dtype=torch.float32) for sample in padded_batch]
        ) for i, key in enumerate(keys)
    }

    # Return the new collate
    return collated


        # Retrieve the distorted EPI image and fieldmap image
        # b0u_img = b0_u[0]
        # fieldmap_img = fieldmap[0]

        # # Check if the fieldmap and distorted epi image has similar dimensions
        # if fieldmap_img.shape != b0u_img.shape:
        #     # If the shapes doesn't fit, resample fieldmap to epi space
        #     # Retrieve the nifti images
        #     b0u_nifti = data_util.get_nifti_image(b0u_img, b0u_affine)
        #     fieldmap_nifti = data_util.get_nifti_image(fieldmap_img, fieldmap_affine)

        #     # Resample the fieldmap image
        #     fieldmap_nifti_resampled = data_util.resample_image(fieldmap_nifti, b0u_nifti.affine, b0u_nifti.shape)

        #     # Get Resampled image data again
        #     fieldmap_nifti_resampled_data = fieldmap_nifti_resampled.get_fdata()
        #     fieldmap_resampled_img = torch.tensor(fieldmap_nifti_resampled_data, dtype=torch.float32).unsqueeze(0)
        # else:
        #     fieldmap_resampled_img = fieldmap

            # print(f"\nb0d_img shape: {b0d_img.shape}")
            # print(f"fieldmap img shape: {fieldmap_img.shape}")
            # print(f"fieldmap resampled img shape: {fieldmap_resampled_img.shape}")


"""
Function for creating the dataloaders
"""
def create_dataloaders(batch_size: int):
    # Torch datasets root
    DATASETS_SAVE_ROOT = "/student/magnuschristensen/dev/fmdc/torch-datasets"

    # Create the paths
    TRAIN_PATH = path.join(DATASETS_SAVE_ROOT, "train_dataset.pt")
    VALIDATION_PATH = path.join(DATASETS_SAVE_ROOT, "val_dataset.pt")
    TEST_PATH = path.join(DATASETS_SAVE_ROOT, "test_dataset.pt")

    # Load the datasets
    print(f"\nLoading .pt datasets...")
    train_dataset = torch.load(TRAIN_PATH, weights_only=False)
    val_dataset = torch.load(VALIDATION_PATH, weights_only=False)
    test_dataset = torch.load(TEST_PATH, weights_only=False)
    print(f".pt datasets loaded")

    # Create the dataloaders
    print(f"\nCreating dataloaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True, collate_fn=collate_fn, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, shuffle=False, collate_fn=collate_fn, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=False, collate_fn=collate_fn, persistent_workers=True)
    print(f"Dataloaders created... returning...")

    # Returning
    return train_dataloader, val_dataloader, test_dataloader

    # def train_dataloader(self):
    #     return DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, num_workers=2, shuffle=True, collate_fn=collate_fn, persistent_workers=True)

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.BATCH_SIZE, num_workers=1, shuffle=False, collate_fn=collate_fn, persistent_workers=True)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.BATCH_SIZE, num_workers=0, shuffle=False, collate_fn=collate_fn, persistent_workers=True)

