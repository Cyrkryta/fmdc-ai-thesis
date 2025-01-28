import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import volumentations
from torch.utils.data import DataLoader, Dataset

from data import fmri_data_util
from data import augmentations


class FMRIDataset(Dataset):

    def __init__(self, subject_paths, device, augment=False, split_temporally=True):
        self.transform = augmentations.Compose([
            augmentations.Rotate((-15, 15), (-15, 15), (-15, 15)),
            augmentations.ElasticTransform((0, 0.25), interpolation=2),
            augmentations.RandomScale([0.9, 1.1], interpolation=1),
            augmentations.Resize((36, 64, 64), interpolation=1, resize_type=0)
        ])
        self.device = device
        self.augment = augment

        self.img_t1 = []
        self.img_b0_d = []
        self.img_b0_u = []
        self.img_mask = []
        self.img_fieldmap = []
        self.affine = []
        self.echo_spacing = []

        for subject_path in subject_paths:
            img_t1, img_b0_d, img_b0_u, img_mask, img_fieldmap, b0u_affine, _, echo_spacing = fmri_data_util.load_data_from_path(subject_path)

            if split_temporally:
                # Split fMRI images along temporal axis into independent data samples
                self.img_t1.extend(list(img_t1))
                self.img_b0_d.extend(list(img_b0_d))
                self.img_b0_u.extend(list(img_b0_u))
                self.img_mask.extend(list(img_mask))
                self.img_fieldmap.extend(list(img_fieldmap))
                self.affine.extend(list(b0u_affine))
                self.echo_spacing.extend(list(echo_spacing))
            else:
                self.img_t1.append(img_t1)
                self.img_b0_d.append(img_b0_d)
                self.img_b0_u.append(img_b0_u)
                self.img_mask.append(img_mask)
                self.img_fieldmap.append(img_fieldmap)
                self.affine.append(b0u_affine)
                self.echo_spacing.append(echo_spacing)

    def __len__(self):
        return len(self.img_t1)

    def __getitem__(self, idx):
        data = {
            'b0_d': self.img_b0_d[idx],
            'b0_u': self.img_b0_u[idx],
            't1': self.img_t1[idx],
            'mask': self.img_mask[idx],
            'fieldmap': self.img_fieldmap[idx]
        }

        if self.augment:
            transformed_data = self.transform(data)
        else:
            transformed_data = data

        img_data = torch.stack((torch.from_numpy(transformed_data['b0_d']).float().to(self.device), torch.from_numpy(transformed_data['t1']).float().to(self.device)))
        return (img_data,
                torch.from_numpy(transformed_data['b0_u']).float().to(self.device),
                torch.from_numpy(transformed_data['mask']).bool().to(self.device),
                torch.from_numpy(transformed_data['fieldmap']).float().to(self.device),
                torch.from_numpy(self.affine[idx]).float().to(self.device),
                torch.from_numpy(np.asarray(self.echo_spacing[idx])).float().to(self.device))


class FMRIDataModule(pl.LightningDataModule):

    def __init__(self, dataset_paths, batch_size, device):
        super(FMRIDataModule).__init__()
        self.dataset_paths = dataset_paths
        self.device = device
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.batch_size = batch_size
        self.load_data()

    def load_data(self):
        subject_paths = fmri_data_util.collect_all_subject_paths(dataset_paths=self.dataset_paths)

        total_count = len(subject_paths)
        train_count = int(0.7 * total_count)
        val_count = int(0.2 * total_count)
        test_count = total_count - train_count - val_count

        rng = torch.Generator()
        rng.manual_seed(0)
        train_paths, val_paths, test_paths = torch.utils.data.random_split(
            subject_paths, (train_count, val_count, test_count),
            generator=rng
        )

        self.train_dataset = FMRIDataset(subject_paths=train_paths, device=self.device, augment=False)  # Disable augmentations for now
        self.val_dataset = FMRIDataset(subject_paths=val_paths, device=self.device, augment=False)
        self.test_dataset = FMRIDataset(subject_paths=test_paths, device=self.device, augment=False)

        self.metrics_dataset = FMRIDataset(subject_paths=val_paths, device=self.device, augment=False, split_temporally=False)

        print(f'Loaded datasets: {len(self.train_dataset)} training samples, {len(self.val_dataset)} validation samples, {len(self.test_dataset)} test samples.')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=1, shuffle=False, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0, shuffle=False, persistent_workers=True)

    def train_val_dataloader(self):
        return DataLoader(self.train_val_dataset, batch_size=self.batch_size, num_workers=2, shuffle=True, persistent_workers=True)

    def metrics_dataloader(self):
        return DataLoader(self.metrics_dataset, batch_size=1, num_workers=1, shuffle=False, persistent_workers=True)
