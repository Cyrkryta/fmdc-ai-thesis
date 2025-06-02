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
# from models.unet3d_fieldmap_crop import UNet3DFieldmap
import json


if __name__ == '__main__':
    """
    Main section for running the computations
    """
    # Defining all of the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, help="Directory for model checkpoints")
    parser.add_argument("--max_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="The desired batch size")
    args = parser.parse_args()

    print(f"Checkpoint path: {args.checkpoint_path}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Batch size: {args.batch_size}")

    # Setting up the device
    device = "cpu"
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method("spawn")
        device = "cuda"
    print(f"Running on {device}")

    train_paths_json = "/indirect/student/magnuschristensen/dev/fmdc/data-paths/training/train_paths_AP.json"
    val_paths_json = "/indirect/student/magnuschristensen/dev/fmdc/data-paths/training/val_paths_AP.json"

    # Load the json files for test
    with open(train_paths_json, "r") as f:
        train_json = json.load(f) 

    with open(val_paths_json, "r") as f:
        val_json = json.load(f) 

    TRAIN_PATHS = train_json["train_paths"]
    VAL_PATHS = val_json["val_paths"]

    print(f"# Number of TRAIN samples: {len(TRAIN_PATHS)}")
    print(f"# Number of VAL samples: {len(VAL_PATHS)}")

    # Setting up the model parameters
    wandb_run = wandb.init(project="field-map-ai", reinit=True)
    wandb_logger = WandbLogger(project="field-map-ai", id=wandb_run.id, log_model=True)
    checkpoint_prefix = f"{wandb_run.id}_model_AP-ONLY_"
    # every_n_epochs = 10
    every_n_epochs = 1
    val_every_n_epoch = 1
    log_every_n_steps = 50
    print(f"\nUpdating model every {every_n_epochs} epochs")
    print(f"Computes validation loss and logs to wandb every {val_every_n_epoch} epoch")
    print(f"Logs for graph every {log_every_n_steps} steps\n")

    # Setting up the checkpoint callback
    print("Setting up checkpoint callback...")
    checkpoint_callback = ModelCheckpoint(
        dirpath = args.checkpoint_path,
        filename = checkpoint_prefix + "unet3d_{epoch:02d}_{val_loss:.5f}",
        every_n_epochs = every_n_epochs,
        save_top_k = 1,
        monitor = "val_loss"
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
        max_epochs = args.max_epochs,
        log_every_n_steps = log_every_n_steps,
        callbacks = [checkpoint_callback, early_stop_callback],
        default_root_dir = args.checkpoint_path,
        check_val_every_n_epoch = val_every_n_epoch,
        logger = wandb_logger
    )

    print("Trainer set up\n")

    print("Creating training dataset....")
    train_dataset = LazyFMRIDataset(TRAIN_PATHS, device=device, mode="train")
    train_sampler = FMRICustomSampler(train_dataset, batch_size=args.batch_size, key_fn=get_project_key)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=2, persistent_workers=True, pin_memory=True)
    print("Training dataset created\n")

    print("Creating validation dataset...")
    val_dataset = LazyFMRIDataset(VAL_PATHS, device=device, mode="train")
    val_sampler = FMRICustomSampler(val_dataset, batch_size=args.batch_size, key_fn=get_project_key)
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
