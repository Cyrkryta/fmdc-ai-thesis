# Import the dependencies
import argparse
import glob
import os.path
import numpy as np
import torch
import pytorch_lightning as L
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
from project.data.fmri_dataset import FMRIDataModule, FMRIDataset
from models.unet3d_fieldmap import UNet3DFieldmap
import inference
from project.data.dataloader import create_dataloaders
from metrics.metrics import TemporalCorrelation
from torchsummary import summary

"""
Main function for doing the training
"""
if __name__ == '__main__':
    # Set up parser and arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--CHECKPOINT_PATH", required=True, help="Path to hold the model checkpoint")
    parser.add_argument("--max_epochs", type=int, default=3, help="Epochs for the model to run")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to use in the model")

    # parse and load the arguments
    args = parser.parse_args()
    CHECKPOINT_PATH = args.CHECKPOINT_PATH
    max_epochs = args.max_epochs
    batch_size = args.batch_size

    # Set the device
    device = 'cpu'
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn')
        device = 'cuda'
        print(f'Running on {device}!')
    
    # Retrieve the dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(batch_size=batch_size)


    # Initialize the data module and 3D unet model
    # data_module = FMRIDataModule(TRAIN_DATASET_PATHS=TRAINING_DATASET_PATHS, BATCH_SIZE=batch_size, device=device, TEST_DATASET_PATHS=TEST_DATASET_PATHS)

    #################### TESTING START ###############
    # Testing the loaders
    sample_batch = next(iter(train_dataloader))

    # Inspecting the batch
    print(f"\nKeys in the batch: {sample_batch.keys()}")
    img_data = sample_batch["img_data"]
    img_data_batch = img_data[0]
    t1 = img_data_batch[0]
    b0 = img_data_batch[1]
    print(f"Img data T1w: value shape={t1.shape}, type={type(t1)}")
    print(f"Img data b0d: value shape={b0.shape}, type={type(b0)}")
    for key, val in sample_batch.items():
        print(f"{key}: value shape={val.shape}, type={type(val)}")

    ################### TESTING END ##################

    print(f"Reached the end right before instantiating the model")

    model = UNet3DFieldmap()
    
    # Connect to weights and biases
    wandb.init(project='field-map-ai')
    wandb_logger = WandbLogger(project='field-map-ai')

    # trianing variables
    checkpoint_prefix = f"{wandb.run.id}_"
    # every_n_epochs = 10
    every_n_epochs = 100
    # val_every_n_epochs = 10
    val_every_n_epochs = 100
    # log_every_n_steps = 10
    log_every_n_steps = 100

    # Define the checkpoint callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,                                            # Output path for checkpoints
        filename=checkpoint_prefix + "unet3d2_{epoch:02d}_{val_loss:.5f}",  # Checkpoint filename structure
        every_n_epochs=every_n_epochs,                                      # How often should a checkpoint be saved
        save_top_k=1,                                                       # Only save the best ckpt (override if better)
        monitor='val_loss',                                                 # Monitors val_loss: lower the better
        save_last=True                                                      # Always save the last model
    )

    # Set up early stopping 
    early_stop_callback = EarlyStopping(monitor='val_loss', mode='min', min_delta=10, patience=5)

    print(f"This is right before defining the trainer, skrt...")

    # Define the trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,                                              # Max number of allowed epochs
        log_every_n_steps=log_every_n_steps,                                # How often is the training loss computed and logged
        callbacks=[checkpoint_callback, early_stop_callback],               # Callback to determine if a model should be saved and training stopped
        default_root_dir=CHECKPOINT_PATH,                                   # Output paths for checkpoints
        check_val_every_n_epoch=val_every_n_epochs,                         # How often the validation loop is run and compute metrics
        logger=wandb_logger                                                 # Logging endpoint
    )

    print("Wow, I reached the end right before the actual training! Nicely done")

    # Train the model. Skrt skrt. 
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
"""
    # Perform inference and evaluation of the trained model
    '''_infer_sample(
        model_path=checkpoint_callback.best_model_path,
        training_dataset_path=args.training_dataset_path,
        device=device
    )

    unet3d.eval()
    unet3d.to(device)

    _evaluate_temporal_correlation(unet3d, data_module.metrics_dataloader())'''
"""

# def _infer_sample(model_path, training_dataset_path, device):
#     """
#     Perform inference for a single sample and log the results to wandb.
#     :param model_path: Path to the model checkpoint that should be used for inference.
#     :param training_dataset_path: Path to the dataset from which the sample should be extracted.
#     :param device: Device on which inference should be performed.
#     """

#     output_path = os.path.join(wandb.run.dir, 'inferred-data')

#     input_dataset_path = glob.glob(training_dataset_path)[0]
#     input_subject_path = glob.glob(os.path.join(input_dataset_path, 'sub-*'))[0]

#     inference.infer_and_store(
#         input_subject_path=input_subject_path,
#         output_path=output_path,
#         checkpoint_path=model_path,
#         device=device
#     )

#     artifact = wandb.Artifact(name='inferred-data', type='model-output')
#     artifact.add_dir(local_path=output_path)
#     wandb.log_artifact(artifact)


# def _evaluate_temporal_correlation(model, dataloader):
#     """
#     Calculate the temporal correlation metric for a trained model and log it to wandb.
#     :param model: The model that should be evaluated.
#     :param dataloader: The dataloader to be used for evaluation, will most likely be a validation or test data loader.
#     """

#     pearson_coefficients_out = []
#     pearson_coefficients_distorted = []

#     print("Calculating temporal correlation...")

#     for batch in tqdm(dataloader):
#         temporal_correlation_out = TemporalCorrelation()
#         temporal_correlation_distorted = TemporalCorrelation()

#         for idx in range(len(batch[1][0])):
#             img = np.swapaxes(batch[0][0], 0, 1)[idx]
#             b0u = batch[1][0][idx]
#             b0d = batch[0][0][0][idx]
#             mask = batch[2][0][idx]

#             # TODO: undistort with nipype here!

#             out = model(img.unsqueeze(0))
#             temporal_correlation_out.update(
#                 ground_truth=b0u.squeeze().detach().cpu().numpy(),
#                 image=torch.where(mask, out, -1).squeeze().detach().cpu().numpy()
#             )
#             temporal_correlation_distorted.update(
#                 ground_truth=b0u.squeeze().detach().cpu().numpy(),
#                 image=b0d.detach().cpu().numpy()
#             )

#         sample_pearson_coefficients_out, _ = temporal_correlation_out.compute()
#         sample_pearson_coefficients_distorted, _ = temporal_correlation_distorted.compute()

#         pearson_coefficients_out.append(sample_pearson_coefficients_out)
#         pearson_coefficients_distorted.append(sample_pearson_coefficients_distorted)

#     wandb.log({
#         'correlation_mean': np.nanmean(np.array(pearson_coefficients_out)),
#         'correlation_std': np.nanstd(np.array(pearson_coefficients_out)),
#         'correlation_median': np.nanmedian(np.array(pearson_coefficients_out))
#     })