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

import inference
from data import fmri_data_util
from data.fmri_dataset import FMRIDataset
from metrics.metrics import TemporalCorrelation
from models.unet3d_fieldmap import UNet3DFieldmap


def _infer_sample(model_path, training_dataset_path, device):
    """
    Perform inference for a single sample and log the results to wandb.
    :param model_path: Path to the model checkpoint that should be used for inference.
    :param training_dataset_path: Path to the dataset from which the sample should be extracted.
    :param device: Device on which inference should be performed.
    """

    output_path = os.path.join(wandb.run.dir, 'inferred-data')

    input_dataset_path = glob.glob(training_dataset_path)[0]
    input_subject_path = glob.glob(os.path.join(input_dataset_path, 'sub-*'))[0]

    inference.infer_and_store(
        input_subject_path=input_subject_path,
        output_path=output_path,
        checkpoint_path=model_path,
        device=device
    )

    artifact = wandb.Artifact(name='inferred-data', type='model-output')
    artifact.add_dir(local_path=output_path)
    wandb.log_artifact(artifact)


def _evaluate_temporal_correlation(model, dataloader):
    """
    Calculate the temporal correlation metric for a trained model and log it to wandb.
    :param model: The model that should be evaluated.
    :param dataloader: The dataloader to be used for evaluation, will most likely be a validation or test data loader.
    """

    pearson_coefficients_out = []
    pearson_coefficients_distorted = []

    print("Calculating temporal correlation...")

    for batch in tqdm(dataloader):
        temporal_correlation_out = TemporalCorrelation()
        temporal_correlation_distorted = TemporalCorrelation()

        for idx in range(len(batch[1][0])):
            img = np.swapaxes(batch[0][0], 0, 1)[idx]
            b0u = batch[1][0][idx]
            b0d = batch[0][0][0][idx]
            mask = batch[2][0][idx]

            # TODO: undistort with nipype here!

            out = model(img.unsqueeze(0))
            temporal_correlation_out.update(
                ground_truth=b0u.squeeze().detach().cpu().numpy(),
                image=torch.where(mask, out, -1).squeeze().detach().cpu().numpy()
            )
            temporal_correlation_distorted.update(
                ground_truth=b0u.squeeze().detach().cpu().numpy(),
                image=b0d.detach().cpu().numpy()
            )

        sample_pearson_coefficients_out, _ = temporal_correlation_out.compute()
        sample_pearson_coefficients_distorted, _ = temporal_correlation_distorted.compute()

        pearson_coefficients_out.append(sample_pearson_coefficients_out)
        pearson_coefficients_distorted.append(sample_pearson_coefficients_distorted)

    wandb.log({
        'correlation_mean': np.nanmean(np.array(pearson_coefficients_out)),
        'correlation_std': np.nanstd(np.array(pearson_coefficients_out)),
        'correlation_median': np.nanmedian(np.array(pearson_coefficients_out))
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_dataset_path', default='/Users/jan/Downloads/openneuro-datasets/preprocessed/ds*/')
    parser.add_argument('--checkpoint_path', default='/Users/jan/Downloads/fmri-ckpts')
    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = 'cpu'
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn')
        device = 'cuda'

    print(f'Running on {device}!')

    dataset_paths = fmri_data_util.collect_all_subject_paths(dataset_paths=glob.glob(args.training_dataset_path))
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_paths)):
        train_paths = [dataset_paths[index] for index in train_idx]
        val_paths = [dataset_paths[index] for index in val_idx]

        print(f"Training fold {fold}!")

        unet3d = UNet3DFieldmap()

        wandb.init(project='field-map-ai')
        wandb_logger = WandbLogger(project='field-map-ai')

        val_every_n_epochs = 100

        checkpoint_prefix = f"{wandb.run.id}_"
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_path,
            filename=checkpoint_prefix + "unet3d2_{epoch:02d}_{val_loss:.5f}",
            every_n_epochs=val_every_n_epochs,
            save_top_k=1,
            monitor='val_loss'
        )

        # This early stopping configuration is only valid for the fieldmap model variant
        early_stop_callback = EarlyStopping(monitor='val_loss', mode='min', min_delta=10, patience=5)

        trainer = L.Trainer(
            max_epochs=args.max_epochs,
            log_every_n_steps=100,
            callbacks=[checkpoint_callback, early_stop_callback],
            default_root_dir=args.checkpoint_path,
            check_val_every_n_epoch=val_every_n_epochs,
            logger=wandb_logger
        )

        train_dataset = FMRIDataset(subject_paths=train_paths, device=device, augment=False)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True, persistent_workers=True)

        val_dataset = FMRIDataset(subject_paths=val_paths, device=device, augment=False)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=1, shuffle=False, persistent_workers=True)

        trainer.fit(
            model=unet3d,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

        wandb.finish()

        # Perform inference and evaluation of the trained model
        '''_infer_sample(
            model_path=checkpoint_callback.best_model_path,
            training_dataset_path=args.training_dataset_path,
            device=device
        )
    
        unet3d.eval()
        unet3d.to(device)
    
        _evaluate_temporal_correlation(unet3d, data_module.metrics_dataloader())'''
