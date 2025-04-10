# Import dependencies
import argparse
import glob
import os.path
import numpy as np
import pytorch_lightning as L
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
import inference
from data import fmri_data_util
from data.fmri_dataset import FMRIDataset
from project.data.lazy_fmri_dataset import LazyFMRIDataset
from project.data.custom_fmri_sampler import FMRICustomSampler, get_project_key
from metrics.metrics import TemporalCorrelation
from models.unet3d_fieldmap import UNet3DFieldmap

def get_trained_folds(checkpoint_dir):
    """
    Function that checks for folds that are already trained
    Done before trained to accommodate potential breaks
    """
    completed_folds = set()
    pattern = re.compile(r"model(\d+)_") # Matches "model{fold}_"
    for filename in os.listdir(checkpoint_dir):
        match = pattern.search(filename)
        if match:
            completed_folds.add(int(match.group(1)))
    return completed_folds

if __name__ == '__main__':
    """
    Main section for running the computations
    """
    # Defining all of the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--CHECKPOINT_PATH", required=True, help="Directory for model checkpoints")
    parser.add_argument("--TRAINING_DATASET_PATH", required=True, help="Path to the training data")
    parser.add_argument("--max_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="The desired batch size")
    args = parser.parse_args()
    CHECKPOINT_PATH = args.CHECKPOINT_PATH
    TRAINING_DATASET_PATH = args.TRAINING_DATASET_PATH
    max_epochs = args.max_epochs
    batch_size = args.batch_size

    # Print the provided variables
    print(f"Training dataset path: {TRAINING_DATASET_PATH}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")
    print(f"Max epochs: {max_epochs}")
    print(f"Batch Size: {batch_size}")

    # Setting up the device
    device = "cpu"
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method("spawn")
        device = "cuda"
    print(f"Running on {device}")

    print(f"Retrieving completed folds...")
    completed_folds = get_trained_folds(CHECKPOINT_PATH)
    print(f"Completed folds: {completed_folds}")

    # Collect all of the paths for training
    DATASET_PATHS = fmri_data_util.collect_all_subject_paths(dataset_paths=glob.glob(TRAINING_DATASET_PATH))
    k_fold = KFold(n_splits=10, shuffle=True, random_state=42)

    # Train each of the folds
    for fold, (train_idx, val_idx) in enumerate(k_fold.split(DATASET_PATHS)):
        # Check if the fold is already trained. Continue if True
        if fold in completed_folds:
            print(f"Fold {fold} has already been trained, skipping...")
            continue

        # Retrieve the paths for training and validation
        TRAIN_PATHS = [DATASET_PATHS[index] for index in train_idx]
        VAL_PATHS = [DATASET_PATHS[index] for index in val_idx]
        print(f"Training fold {fold}")

        # Setting up parameters for training
        wandb_run = wandb.init(project="field-map-ai", reinit=True)
        wandb_logger = WandbLogger(project="field-map-ai", id=wandb_run.id, log_model=True)
        checkpoint_prefix = f"{wandb_run.id}_model{fold}_"
        every_n_epochs = 10
        val_every_n_epoch = 1
        log_every_n_steps = 50
        print(f"\nUpdating model every {every_n_epochs} epochs")
        print(f"Computes validation loss and logs to wandb every {val_every_n_epoch} epoch")
        print(f"Logs for graph every {log_every_n_steps} steps\n")

        # Setting up the checkpoint callback
        print("Setting up checkpoint callback...")
        checkpoint_callback = ModelCheckpoint(
            dirpath = CHECKPOINT_PATH,
            filename = checkpoint_prefix + "unet3d_{epoch:02d}_{val_loss:.5f}",
            every_n_epochs = every_n_epochs,
            save_top_k = 1,
            monitor = "val_loss",
            save_last=True
        )
        print("Checkpoint callback set up\n")
        
        print("Setting up early stopping callback")
        early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", min_delta=10, patience=5)
        print("Early stopping callback set up\n")

        print("Setting up model...")
        model = UNet3DFieldmap()
        print("Model set up...\n")

        print("Setting up Trainer...")
        trainer = L.Trainer(
            max_epochs = max_epochs,
            log_every_n_steps = log_every_n_steps,
            callbacks = [checkpoint_callback, early_stop_callback],
            default_root_dir = CHECKPOINT_PATH,
            check_val_every_n_epoch = val_every_n_epoch,
            logger = wandb_logger
        )
        print("Trainer set up\n")

        print("Creating training dataset....")
        train_dataset = LazyFMRIDataset(TRAIN_PATHS, device=device, mode="train")
        train_sampler = FMRICustomSampler(train_dataset, batch_size=batch_size, key_fn=get_project_key)
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=2, persistent_workers=True)
        print("Training dataset created\n")

        print("Creating validation dataset...")
        val_dataset = LazyFMRIDataset(VAL_PATHS, device=device, mode="train")
        val_sampler = FMRICustomSampler(val_dataset, batch_size=batch_size, key_fn=get_project_key)
        val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=1, persistent_workers=True)
        print("Validation dataset created\n")

        # Training the model
        print(f"Training model...")
        trainer.fit(
            model = model,
            train_dataloaders = train_dataloader,
            val_dataloaders = val_dataloader
        )

        # Finish wandb
        wandb.finish()
