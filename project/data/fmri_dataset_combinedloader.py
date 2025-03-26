import os
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning.trainer.supporters import CombinedLoader
from project.data import fmri_data_util
import torch
from project.data import augmentations
import numpy as np
import pytorch_lightning as pl


class FMRIDataset(Dataset):
    """
    Class for creating the data as intended
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

class FMRIDataModule(pl.LightningDataModule):
    """
    Module for creating the datasets
    """
    def __init__(self, TRAIN_DATASET_PATHS, DATASET_SAVE_ROOT, device):
        super().__init__()
        self.TRAIN_DATASET_PATHS = TRAIN_DATASET_PATHS
        self.DATASET_SAVE_ROOT = DATASET_SAVE_ROOT
        self.device = device
        self.train_datasets = {}  # Dictionary: {project_name: dataset}
        self.val_datasets = {}    # Dictionary: {project_name: dataset}
        self.test_dataset = None  # If needed later
        self.load_data()

    def load_data(self):
        # Collect all subject paths from the training datasets
        all_subject_paths = fmri_data_util.collect_all_subject_paths(dataset_paths=self.TRAIN_DATASET_PATHS)
        
        # Group subjects by project
        grouped = {}
        for subject_path in all_subject_paths:
            # Example: subject_path = /.../trainval-processed/ds002422/sub-01
            project = os.path.basename(os.path.dirname(subject_path))
            grouped.setdefault(project, []).append(subject_path)

        # For each of the projects, split into training and validation (80/20)
        rng = torch.Generator()
        rng.manual_seed(0)
        # Split each dataset into train and validation
        for proj, paths in grouped.items():
            total_size = len(paths)
            train_size = int(0.8 * total_size)
            val_size = total_size - train_size
            train_split, val_split = random_split(paths, [train_size, val_size], generator=rng)
            # Create the fMRI datasets for each split
            # Train split
            self.train_datasets[proj] = FMRIDataset(SUBJECT_PATHS=list(train_split), device=self.device, augment=False, split_temporally=True)
            # Val split
            self.val_datasets[proj] =  FMRIDataset(SUBJECT_PATHS=list(val_split), device=self.device, augment=False, split_temporally=True)

        # Print the dataset
        print(f"Grouped projects: {list(self.train_datasets.keys())}")
        total_train_samples = sum(len(ds) for ds in self.train_datasets.values())
        total_val_samples = sum(len(ds) for ds in self.val_datasets.values())
        print(f"Loaded datasets: {total_train_samples} training samples, {total_val_samples} validation samples")
    
    # Defining the train dataloader
    def train_dataloader(self):
        # Create a single dataloader per projects
        loaders = {proj: DataLoader(ds, batch_size=32, shuffle=True, num_workers=2) for proj, ds in self.train_datasets.items()}
        # Create combined pytorch lightning dataloader
        combined_train_loader = CombinedLoader(loaders, mode="max_size_cycle")
        # Return the combined dataloader
        return combined_train_loader
    
    # Defining the val dataloader
    def val_dataloader(self):
        # Create a single dataloader per project
        loaders = {proj: DataLoader(ds, batch_siz=32, shuffle = True, num_workers=1) for proj, ds in self.val_datasets.items()}
        # Create a combined pytorch lightning dataloader
        combined_val_loader = CombinedLoader(loaders, mode="max_size_cycle")
        # Return the combined dataloader
        return combined_val_loader

