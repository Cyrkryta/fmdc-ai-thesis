# Importing all the dependencies
import time
import numpy as np
import torch
from tqdm import tqdm
import os
import nibabel as nib
from skimage import metrics
from project.metrics.metrics import TemporalCorrelation

"""
Class:
Compute the metrics for a particular object
"""
class MetricsComputationBase(object):

    # Initialize the class
    def __init__(self):
        self.inference_compute_times = []

    # Get all the subject paths
    def get_subject_paths(self):
        raise NotImplementedError()

    # Load the data from a particular sample
    def load_input_samples(self, subject_path):
        raise NotImplementedError()

    # Get the undistorted output of the sample
    def get_undistorted_b0(self, sample):
        raise NotImplementedError()

    # Compute the metrics
    def compute_metrics(self):
        # Metric placeholders to be computed
        metric_values = {
            'correlation_distorted_mean': [],
            'correlation_out_mean': [],
            'correlation_distorted_median': [],
            'correlation_out_median': [],
            'mse_distorted_mean': [],
            'mse_out_mean': [],
            'mse_distorted_median': [],
            'mse_out_median': [],
            'ssim_distorted_mean': [],
            'ssim_out_mean': [],
            'ssim_distorted_median': [],
            'ssim_out_median': []
        }

        # Go through each of the subjects
        for subject_path in tqdm(self.get_subject_paths()):
            # Compute the distorted and undistorted temporal correlation
            temporal_correlation_out = TemporalCorrelation()
            temporal_correlation_distorted = TemporalCorrelation()
            # MSE losses distorted and undistorted
            mse_losses_out = []
            mse_losses_distorted = []
            # SSIM distorted and undistorted
            ssim_out = []
            ssim_distorted = []
            # affine = None

            # Load the samples for a subject
            for sample in self.load_input_samples(subject_path):
                # print(sample)
                # Start time to figure how long it takes
                start = time.time()
                # Get the computed undistorted output
                undistorted_b0, fieldmap_out = self.get_undistorted_b0(sample)
                # End the time and add to the computation times
                end = time.time()
                self.inference_compute_times.append(end - start)

                # Retrieved the affined image for b0 and fieldmap
                # ... Why though??? This is kinda stupid
                # affine_b0 = sample['b0u_affine']
                # affine_fm = sample['fieldmap_affine']

                # Retrieve the temporal correlation between undistorted gt and estimated (mask) from fieldmap
                temporal_correlation_out.update(
                    ground_truth=sample['b0u'].squeeze(),
                    image=np.where(sample['mask'], undistorted_b0, -1).squeeze()
                )

                # Retrieve the temporal correlation between undistorted and distorted ground truths
                temporal_correlation_distorted.update(
                    ground_truth=sample['b0u'].squeeze(),
                    image=sample['b0d']
                )

                # Compute mse loss between undistorted gt and estimation from fieldmap
                mse_losses_out.append(np.square(np.subtract(
                    np.where(sample['mask'], undistorted_b0, -1).squeeze(),
                    sample['b0u'].squeeze()
                )).mean())

                # Compute mse loss between undistorted and distorted ground truths
                mse_losses_distorted.append(np.square(np.subtract(
                    sample['b0d'].squeeze(),
                    sample['b0u'].squeeze()
                )).mean())

                # Stack all of the images together within the mask and compute the data range
                all_imgs = np.stack([np.where(sample['mask'], undistorted_b0, -1).squeeze(), sample['b0u'].squeeze(), sample['b0d'].squeeze()])
                data_range = np.max(all_imgs) - np.min(all_imgs)
                
                # Compute SSIM betweeen gt undistorted and estimated undistorted
                ssim_out.append(metrics.structural_similarity(
                    np.where(sample['mask'], undistorted_b0, -1).squeeze(),
                    sample['b0u'].squeeze(),
                    data_range=data_range
                ))

                # Compute SSIM between gt distorted and undistorted
                ssim_distorted.append(metrics.structural_similarity(
                    sample['b0d'].squeeze(),
                    sample['b0u'].squeeze(),
                    data_range=data_range
                ))
                
            # Compute the pearons coefficient, output mse loss and outputs ssim
            pearson_coefficients_out, pearson_p_out = temporal_correlation_out.compute()
            pearson_coefficients_distorted, pearson_p_distorted = temporal_correlation_distorted.compute()
            mse_losses_out = np.array(mse_losses_out)
            mse_losses_distorted = np.array(mse_losses_distorted)
            ssim_out = np.array(ssim_out)
            ssim_distorted = np.array(ssim_distorted)

            # Report everything
            print(f'Median distorted={np.nanmedian(pearson_coefficients_distorted.flatten())} vs. Median out={np.nanmedian(pearson_coefficients_out.flatten())} ({subject_path})')
            print(f'Mean distorted={np.nanmean(pearson_coefficients_distorted.flatten())} vs. Mean out={np.nanmean(pearson_coefficients_out.flatten())} ({subject_path})')
            print(f'Mean MSE distorted={mse_losses_distorted.mean()} vs. Mean MSE out={mse_losses_out.mean()}')
            print(f'Mean SSIM distorted={ssim_distorted.mean()} vs. Mean SSIM out={ssim_out.mean()}')

            metric_values['correlation_distorted_mean'].append(np.nanmean(pearson_coefficients_distorted.flatten()))
            metric_values['correlation_out_mean'].append(np.nanmean(pearson_coefficients_out.flatten()))
            metric_values['correlation_distorted_median'].append(np.nanmedian(pearson_coefficients_distorted.flatten()))
            metric_values['correlation_out_median'].append(np.nanmedian(pearson_coefficients_out.flatten()))
            metric_values['mse_distorted_mean'].append(mse_losses_distorted.mean())
            metric_values['mse_out_mean'].append(mse_losses_out.mean())
            metric_values['mse_distorted_median'].append(np.median(mse_losses_distorted))
            metric_values['mse_out_median'].append(np.median(mse_losses_out))
            metric_values['ssim_distorted_mean'].append(ssim_distorted.mean())
            metric_values['ssim_out_mean'].append(ssim_out.mean())
            metric_values['ssim_distorted_median'].append(np.median(ssim_distorted))
            metric_values['ssim_out_median'].append(np.median(ssim_out))

        print('\n\nMETRICS REPORT\n')
        for metric_name, values in metric_values.items():
            print(f'{metric_name}: {values}')
            print(f'Mean {metric_name}: {np.array(values).mean()}')
            print(f'Median {metric_name}: {np.median(np.array(values))}')

        print(f'Mean compute time per sample in seconds: {np.mean(np.array(self.inference_compute_times))}')
