import os
import argparse
import json
import numpy as np
import nibabel as nib
from tqdm import tqdm
from project.metrics.metrics import TemporalCorrelation
from project.metrics_scripts.fieldmaps_model import FieldmapsModelMetricsComputation

class VoxelwiseFieldmapMetricsComputation(FieldmapsModelMetricsComputation):
    """
    Subclass of FieldmapsModelMetricsComputation that, instead of
    summarizing metrics per subject, builds full 3D metric maps.

    Inherits:
      - get_subject_paths(): yields list of subject folders
      - load_input_samples(subject_path): returns per-timepoint samples dicts
      - get_undistorted_b0(sample): runs model + fugue to output undistorted B0
      - Timing lists: inference_compute_times, fieldmap_compute_times, undistortion_compute_times
    """
    def compute_map_metrics(self, out_dir: str, metric: str = 'delta_r'):
        # Ensure output directory exists
        os.makedirs(out_dir, exist_ok=True)
        mask3d = None
        # Run through each subject
        for subj in tqdm(self.TEST_PATHS, desc="Subjects"):
            if metric == "delta_r":
                tc_dist = TemporalCorrelation()
                tc_pred = TemporalCorrelation()
            elif metric == "absdiff":
                absdiffs = []
            else:
                raise ValueError(f"Unsupported metric '{metric}'")

            # Iterate through each timepoint/sample using inherited method
            for sample in self.load_input_samples(subj):
                gt = sample["b0u"].squeeze()                # Ground truth volume
                dist = sample["b0d"].squeeze()              # Distorted volumes
                pred, _ = self.get_undistorted_b0(sample)   # Model corrected output
                mask = sample["mask"].squeeze()
                mask3d = None

                if metric == "delta_r":
                    gt_masked   = np.where(mask, gt,   0)
                    dist_masked = np.where(mask, dist, 0)
                    pred_masked = np.where(mask, pred, 0)
                    tc_dist.update(ground_truth=gt_masked, image=dist_masked)
                    tc_pred.update(ground_truth=gt_masked, image=pred_masked)          

                    # tc_dist.update(ground_truth=gt, image=dist)
                    # pred_masked = pred if mask is None else np.where(mask, pred, 0)
                    # tc_pred.update(ground_truth=gt, image=pred_masked)
                else:
                    absdiffs.append(np.abs(pred - gt))

            # Compute the volumes
            dataset_id = os.path.basename(os.path.dirname(subj.rstrip(os.sep)))
            subject_name  = os.path.basename(subj.rstrip(os.sep))
            subject_id = f"{dataset_id}_{subject_name}"
            if metric == "delta_r":
                r_dist, _ = tc_dist.compute()
                r_pred, _ = tc_pred.compute()
                # print("r_dist: min, max, mean =", np.nanmin(r_dist), np.nanmax(r_dist), np.nanmean(r_dist))
                # print("r_pred: min, max, mean =", np.nanmin(r_pred), np.nanmax(r_pred), np.nanmean(r_pred))
                delta_r = r_pred - r_dist
                delta_r = np.where(mask, delta_r, np.nan)

                # print("delta_r: min, max, mean =", np.nanmin(delta_r), np.nanmax(delta_r), np.nanmean(delta_r))
                img = delta_r.astype(np.float32)

            else:
                # Average absolute difference across time points
                img = np.mean(np.stack(absdiffs), axis=0).astype(np.float32)

            # Retrieve the affine from the ground truth undistorted file
            b0u_file = os.path.join(subj, "b0_u.nii.gz")
            if not os.path.exists(b0u_file):
                raise RuntimeError(f"No b0u NIfTI found in {subj} to extract affine")
            affine = nib.load(b0u_file).affine

            # Save the file to the output
            out_file = os.path.join(out_dir, f"{subject_id}_{metric}.nii.gz")
            nib.save(nib.Nifti1Image(img, affine), out_file)
            print(f"Saved: {out_file}")

def main():
    """
    Main function for running the script
    """
    parser = argparse.ArgumentParser(description="Compute voxelwise metric maps for the fieldmap model")
    parser.add_argument("--test_json", required=True, help="JSON with 'test_paths' listing subject directories")
    parser.add_argument("--metric", choices=['delta_r','absdiff'], default='delta_r', help="Type of voxelwise map to compute")
    parser.add_argument("--out_dir", required=True, help="Directory to save per-subject NIfTI maps")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint for fieldmap inference")
    parser.add_argument("--device", default='cpu', help="Torch device (cpu or cuda)")
    args = parser.parse_args()

    # Retrieve the test paths
    with open(args.test_json, 'r') as f:
        data = json.load(f)
    test_paths = data["test_paths"]
    if not test_paths:
        raise ValueError('No test_paths found in JSON')
    
    # Instantiate the voxel wise computation
    vmc = VoxelwiseFieldmapMetricsComputation(
        CHECKPOINT_PATH=args.checkpoint,
        TEST_PATHS=test_paths,
        device=args.device
    )

    # Compute and write out the maps
    vmc.compute_map_metrics(out_dir=args.out_dir, metric=args.metric)

if __name__ == "__main__":
    main()
