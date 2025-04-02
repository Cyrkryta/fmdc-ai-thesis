import os
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from project.data import fmri_data_util
from project.data.custom_fmri_dataset import FMRIDataset
from project.data.lazy_fmri_dataset import LazyFMRIDataset
from project.data.custom_fmri_sampler import FMRICustomSampler, get_project_key
import sys
import json

def save_to_json(TRAIN_PATHS, VAL_PATHS, TEST_PATHS):
    """
    Function for saving the designated paths
    """
    root = "/student/magnuschristensen/dev/fmdc/data-paths"
    with open(os.path.join(root, "train_paths.json"), "w") as f:
        json.dump({"train_paths": TRAIN_PATHS}, f, indent=4)
    with open(os.path.join(root, "val_paths.json"), "w") as f:
        json.dump({"val_paths": VAL_PATHS}, f, indent=4)
    if TEST_PATHS is not None:
        with open(os.path.join(root, "test_paths.json"), "w") as f:
            json.dump({"test_paths": TEST_PATHS}, f, indent=4)
    print("JSON files for train, validation and test has been saved...")

class FMRIDataModule(pl.LightningDataModule):
    """
    Custom class for creating the FMRI Data Module
    """
    def __init__(self, TRAIN_DATASET_PATHS, DATASET_SAVE_ROOT, device, TEST_DATASET_PATHS=None, batch_size=32):
        super().__init__()
        self.TRAIN_DATASET_PATHS = TRAIN_DATASET_PATHS
        self.TEST_DATASET_PATHS = TEST_DATASET_PATHS
        self.DATASET_SAVE_ROOT = DATASET_SAVE_ROOT
        self.device = device
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        train_save_path = os.path.join(self.DATASET_SAVE_ROOT, "train_dataset_paths.pt")
        val_save_path = os.path.join(self.DATASET_SAVE_ROOT, "val_dataset_paths.pt")
        # test_save_path = os.path.join(self.TEST_DATASET_PATHS, "test_dataset_paths.pt")

        if os.path.exists(train_save_path) and os.path.exists(val_save_path):
            print("Loading pre-saved datasets...")
            self.train_dataset = torch.load(train_save_path, weights_only=False)
            self.val_dataset = torch.load(val_save_path, weights_only=False)
            print(f"loaded {len(self.train_dataset)} training samples and {len(self.val_dataset)} validation samples")
        else:
            print(f"Datasets not found... creating datasets")
            ALL_SUBJECT_PATHS = fmri_data_util.collect_all_subject_paths(dataset_paths=self.TRAIN_DATASET_PATHS)
            total = len(ALL_SUBJECT_PATHS)
            train_count = int(0.9 * total) # 10 percent of total samples for validation
            val_count = total - train_count

            rng = torch.Generator()
            rng.manual_seed(0)
            train_split, val_split = random_split(ALL_SUBJECT_PATHS, [train_count, val_count], generator=rng)

            self.train_dataset = list(train_split)
            self.val_dataset = list(val_split)

            # Creating the separate test dataset
            if self.TEST_DATASET_PATHS is not None:
                TEST_SUBJECT_PATHS = fmri_data_util.collect_all_subject_paths(dataset_paths=self.TEST_DATASET_PATHS)
                self.test_dataset = list(TEST_SUBJECT_PATHS)

            # Save all of the datasets
            os.makedirs(self.DATASET_SAVE_ROOT, exist_ok=True)
            torch.save(self.train_dataset, train_save_path)
            torch.save(self.val_dataset, val_save_path)
            # torch.save(self.test_dataset, test_save_path)

            # Create json files
            save_to_json(TRAIN_PATHS=self.train_dataset, VAL_PATHS=self.val_dataset, TEST_PATHS=self.test_dataset)

            # Exit and prompt
            print("Subject pahts datasets saved to disk...")
            print("Exiting: Please re-run the training script now that the dataset have been created...")
            sys.exit(0)

    def train_dataloader(self):
        from project.data.custom_fmri_sampler import FMRICustomSampler, get_project_key
        dataset = LazyFMRIDataset(self.train_dataset, device=self.device, mode="train")
        sampler = FMRICustomSampler(dataset, batch_size=self.batch_size, key_fn=get_project_key)
        return DataLoader(dataset, batch_sampler=sampler, num_workers=2, persistent_workers=True)
    
    def val_dataloader(self):
        from project.data.custom_fmri_sampler import FMRICustomSampler, get_project_key
        dataset = LazyFMRIDataset(self.val_dataset, device=self.device, mode="train")
        sampler = FMRICustomSampler(dataset, batch_size=self.batch_size, key_fn=get_project_key)
        return DataLoader(dataset, batch_sampler=sampler, num_workers=1, persistent_workers=True)
    
    # def test_dataloader(self):
    #     from project.data.custom_fmri_sampler import FMRICustomSampler, get_project_key
    #     dataset = LazyFMRIDataset(self.test_dataset, device=self.device, mode="test")