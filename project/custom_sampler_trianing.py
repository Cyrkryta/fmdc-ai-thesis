# Importing all of the dependencies
import argparse, glob, os.path, wandb, inference, torch
import numpy as np
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
from project.data.custom_fmri_datamodule import FMRIDataModule
from models.unet3d_fieldmap import UNet3DFieldmap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--CHECKPOINT_PATH", required=True, help="Directory for model checkpoints")
    parser.add_argument("--max_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--TRAINING_DATASET_PATH", required=True, help="Path to the training data")
    parser.add_argument("--DATASET_SAVE_ROOT", required=True, help="Path to datasets that should be loaded")
    parser.add_argument("--batch_size", type=int, default=32, help="The desired batch size")
    args = parser.parse_args()

    CHECKPOINT_PATH = args.CHECKPOINT_PATH
    max_epochs = args.max_epochs
    TRAINING_DATASET_PATHS = glob.glob(args.TRAINING_DATASET_PATH)
    DATASET_SAVE_ROOT = args.DATASET_SAVE_ROOT
    batch_size = args.batch_size

    device = "cpu"
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method("spawn")
        device = "cuda"
        print(f"Running on {device}")

    data_module = FMRIDataModule(
        TRAIN_DATASET_PATHS=TRAINING_DATASET_PATHS,
        DATASET_SAVE_ROOT=DATASET_SAVE_ROOT,
        device=device,
        batch_size=batch_size
    )
    data_module.prepare_data()

    # train_dataloader = data_module.train_dataloader()
    # sample_batch = next(iter(train_dataloader))
    # for key, val in sample_batch.items():
    #     if isinstance(val, list):
    #         print(f"{key}: list length={len(val)}, type={type(val)}")
    #     elif hasattr(val, "shape"):
    #         print(f"{key}: value shape={val.shape}, type={type(val)}")
    #     else:
    #         print(f"{key}: type={type(val)}")


    # # If we get past prepare data, then create the model
    model = UNet3DFieldmap()
    
    wandb.init(project='field-map-ai')
    wandb_logger = WandbLogger(project='field-map-ai')
    
    checkpoint_prefix = f"{wandb.run.id}_"
    every_n_epochs = 100
    val_every_n_epochs = 100
    log_every_n_steps = 100

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        filename=checkpoint_prefix + "unet3d2_{epoch:02d}_{val_loss:.5f}",
        every_n_epochs=100,
        save_top_k=1,
        monitor="val_loss",
        save_last=True
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", min_delta=10, patience=5)

    trainer = L.Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        callbacks=[checkpoint_callback, early_stop_callback],
        default_root=CHECKPOINT_PATH,
        check_val_every_n_epochs=100,
        logger=wandb_logger
    )

    # trainer.fit(
    #     model=model,
    #     train_dataloaders = data_module.train_dataloader(),
    #     val_dataloaders = data_module.val_dataloader()
    # )