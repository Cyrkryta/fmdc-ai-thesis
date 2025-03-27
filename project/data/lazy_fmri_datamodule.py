import os
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from project.data import fmri_data_util
from project.data.custom_fmri_dataset import FMRIDataset
from project.data.lazy_fmri_dataset import LazyFMRIDataset
from project.data.custom_fmri_sampler import FMRICustomSampler, get_project_key
import sys

class FMRIDataModule(pl.LightningDataModule):
    """
    Custom class for creating the FMRI Data Module
    """
    def __init__(self, TRAIN_DATASET_PATHS, DATASET_SAVE_ROOT, device, batch_size=32):
        super().__init__()
        self.TRAIN_DATASET_PATHS = TRAIN_DATASET_PATHS
        self.DATASET_SAVE_ROOT = DATASET_SAVE_ROOT
        self.device = device
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        train_save_path = os.path.join(self.DATASET_SAVE_ROOT, "train_dataset_paths.pt")
        val_save_path = os.path.join(self.DATASET_SAVE_ROOT, "val_dataset_paths.pt")

        if os.path.exists(train_save_path) and os.path.exists(val_save_path):
            print("Loading pre-saved datasets...")
            self.train_dataset = torch.load(train_save_path, weights_only=False)
            self.val_dataset = torch.load(val_save_path, weights_only=False)
            print(f"loaded {len(self.train_dataset)} training samples and {len(self.val_dataset)} validation samples")
        else:
            print(f"Datasets not found... creating datasets")
            ALL_SUBJECT_PATHS = fmri_data_util.collect_all_subject_paths(dataset_paths=self.TRAIN_DATASET_PATHS)
            total = len(ALL_SUBJECT_PATHS)
            train_count = int(0.8 * total)
            val_count = total - train_count

            rng = torch.Generator()
            rng.manual_seed(0)
            train_split, val_split = random_split(ALL_SUBJECT_PATHS, [train_count, val_count], generator=rng)

            self.train_dataset = list(train_split)
            self.val_dataset = list(val_split)
            os.makedirs(self.DATASET_SAVE_ROOT, exist_ok=True)
            torch.save(self.train_dataset, train_save_path)
            torch.save(self.val_dataset, val_save_path)

            print("Subject pahts datasets saved to disk...")
            print("Exiting: Please re-run the training script now that the dataset have been created...")
            sys.exit(0)

    def train_dataloader(self):
        from project.data.custom_fmri_sampler import FMRICustomSampler, get_project_key
        dataset = LazyFMRIDataset(self.train_dataset, device=self.device, mode="train")
        sampler = FMRICustomSampler(dataset, batch_size=self.batch_size, key_fn=get_project_key)
        return DataLoader(dataset, batch_sampler=sampler, num_workers=2, pin_memory=True)
    
    def val_dataloader(self):
        from project.data.custom_fmri_sampler import FMRICustomSampler, get_project_key
        dataset = LazyFMRIDataset(self.val_dataset, device=self.device, mode="train")
        sampler = FMRICustomSampler(dataset, batch_size=self.batch_size, key_fn=get_project_key)
        return DataLoader(dataset, batch_sampler=sampler, num_workers=1, pin_memory=True)