from typing import Any

import numpy as np
from scipy.special import betainc
from torchmetrics import Metric


class TemporalCorrelation(Metric):

    def __init__(self):
        super().__init__()
        self.gt_volumes = []
        self.image_volumes = []
        self.volume_shape = None

    def update(self, ground_truth, image) -> None:
        if self.volume_shape is None:
            self.volume_shape = ground_truth.shape

        if ground_truth.shape != self.volume_shape:
            raise Exception(
                f"Expected volumes of shape {self.volume_shape}, but got ground truth with shape {ground_truth.shape}")
        if image.shape != self.volume_shape:
            raise Exception(f"Expected volumes of shape {self.volume_shape}, but got image with shape {image.shape}")

        self.gt_volumes.append(ground_truth)
        self.image_volumes.append(image)

    def _pearsonr(self, x, y):
        n = x.shape[-1]

        # Compute Pearson correlation coefficient. We can't use `cov` or `corrcoef`
        # because they want to compute everything pairwise between rows of a
        # stacked x and y.
        xm = x.mean(axis=-1, keepdims=True)
        ym = y.mean(axis=-1, keepdims=True)
        cov = np.sum((x - xm) * (y - ym), axis=-1) / (n - 1)
        sx = np.std(x, ddof=1, axis=-1)
        sy = np.std(y, ddof=1, axis=-1)
        rho = cov / (sx * sy)

        # Compute the two-sided p-values. See documentation of scipy.stats.pearsonr.
        ab = n / 2 - 1
        x = (abs(rho) + 1) / 2
        p = 2 * (1 - betainc(ab, ab, x))
        return rho, p

    def _compute_temporal_correlation_matrix(self, a, b):
        a = a.transpose(1, 2, 3, 0)
        b = b.transpose(1, 2, 3, 0)

        # Flatten the spatial dimensions to apply pearsonr across the last dimension
        a_flat = a.reshape(-1, a.shape[-1])
        b_flat = b.reshape(-1, b.shape[-1])

        # Compute Pearson correlation coefficient and p-value for each pair of flattened arrays
        pearson_coefficients = np.empty(a_flat.shape[0])
        pearson_p_values = np.empty(a_flat.shape[0])

        for idx in range(a_flat.shape[0]):
            rho, p = self._pearsonr(a_flat[idx], b_flat[idx])
            pearson_coefficients[idx] = rho
            pearson_p_values[idx] = p

        # Reshape the results back to the original spatial dimensions
        pearson_coefficients = pearson_coefficients.reshape(a.shape[:-1])
        pearson_p_values = pearson_p_values.reshape(a.shape[:-1])

        return pearson_coefficients, pearson_p_values

    def compute(self) -> Any:
        gt_volumes = np.stack(self.gt_volumes)
        image_volumes = np.stack(self.image_volumes)

        # Extract voxel-wise correlations
        pearson_coefficients, pearson_p = self._compute_temporal_correlation_matrix(gt_volumes, image_volumes)

        return pearson_coefficients, pearson_p
