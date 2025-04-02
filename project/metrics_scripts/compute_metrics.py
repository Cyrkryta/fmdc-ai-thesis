# Import the dependencies
import glob
import argparse
import json
from sklearn.model_selection import KFold
from project.data import fmri_data_util
from project.metrics_scripts.baseline_model import BaselineModelMetricsComputation
from project.metrics_scripts.direct_model import DirectModelMetricsComputation
from project.metrics_scripts.fieldmaps_model import FieldmapsModelMetricsComputation
from project.metrics_scripts.k_fold_fieldmaps_model import KFoldFieldmapsModelMetricsComputation
from project.metrics_scripts.mean_fieldmaps import MeanFieldmapsMetricsComputation

"""
Compute all performance metrics on a model
"""
def main(model="fieldmap"):
    # Parse and load arguments
    parser = argparse.ArgumentParser(description="Compute performance metrics for the fieldmap model.")
    parser.add_argument("--CHECKPOINT_PATH", required=True, help="Path to the model checkpoint.")
    parser.add_argument("--TEST_JSON_PATH", required=True, help="Path to the JSON file containing test paths.")
    parser.add_argument("--device", default="cpu", help="Device to run evaluation on (e.g., 'cpu' or 'cuda').")
    args = parser.parse_args()
    CHECKPOINT_PATH = args.CHECKPOINT_PATH
    TEST_JSON_PATH = args.TEST_JSON_PATH
    device = args.device

    # Load the test split from JSONow 
    with open(TEST_JSON_PATH, "r") as f:
        test_data = json.load(f)
    TEST_PATHS=test_data["test_paths"]
    print(f"Processing {len(TEST_PATHS)} test samples")


    if model=="fieldmap":
        # Initialize the metrics computation class with the test paths.
        metrics_comp = FieldmapsModelMetricsComputation(
            CHECKPOINT_PATH=CHECKPOINT_PATH,
            TEST_PATHS=TEST_PATHS,
            device=device
        )
        test_paths = metrics_comp.get_subject_paths()
        print(test_paths)
        print("Skibidi I am right here")
    #     # Compute and print metrics.
        metrics_comp.compute_metrics()

if __name__ == '__main__':
    main()
    # # Define what model to compute
    # mode = 'fieldmaps-mode'

    # # Evaluating the fieldmap model
    # if mode == 'fieldmaps-model':
    #     metrics_computation = FieldmapsModelMetricsComputation(
    #         checkpoint_path='/home/mlc/dev/fmdc/trained-models/jan-models/jan_ruby-sunset-unet3d2_epoch=4799_val_loss=2670.47461.ckpt',
    #         dataset_root='/home/mlc/dev/fmdc/downloads/openneuro-datasets/preprocessed/ds*/',
    #         device=
    #     ).compute_metrics()
    
    # # Evalauting the k-fold fieldmap model
    # elif mode == 'k-fold-fieldmaps-model':
    #     # TODO: Do this with an actual held-out test set instead!
    #     # Setup the dataset and kfold
    #     dataset_paths = fmri_data_util.collect_all_subject_paths(dataset_paths=glob.glob('/Users/jan/Downloads/openneuro-datasets/preprocessed/ds*/'))
    #     kf = KFold(n_splits=10, shuffle=True, random_state=0)

    #     # Define checkpoints to compute folds on
    #     checkpoints_for_folds = [
    #         'd81m39le_unet3d2_epoch=3199_val_loss=2123.65649.ckpt',
    #         'xom4ggju_unet3d2_epoch=3699_val_loss=3404.03809.ckpt',
    #         'x8ysdqfk_unet3d2_epoch=3099_val_loss=2387.13867.ckpt',
    #         'lqhdvqhw_unet3d2_epoch=4199_val_loss=2835.27222.ckpt',
    #         'jzeqtjv7_unet3d2_epoch=3199_val_loss=1815.36096.ckpt',
    #         'm2t3mxvg_unet3d2_epoch=2799_val_loss=2568.07300.ckpt',
    #         'gc0q963r_unet3d2_epoch=2699_val_loss=2184.26831.ckpt',
    #         'soebecyo_unet3d2_epoch=3099_val_loss=2418.63208.ckpt',
    #         'knkd6h4l_unet3d2_epoch=3599_val_loss=2835.74487.ckpt',
    #         '6twf1mhu_unet3d2_epoch=2699_val_loss=2540.18726.ckpt'
    #     ]

    #     # Full path to the checkpoints
    #     checkpoint_paths = [f'/Users/jan/Downloads/k-fold-validation-ckpts/{v}' for v in checkpoints_for_folds]

    #     # Go through each of the folds and compute the metrics
    #     for fold, (_, val_idx) in enumerate(kf.split(dataset_paths)):
    #         val_paths = [dataset_paths[index] for index in val_idx]

    #         print(f"Fold {fold}:")

    #         KFoldFieldmapsModelMetricsComputation(
    #             checkpoint_path=checkpoint_paths[fold],
    #             subject_paths=val_paths,
    #             device='cpu'
    #         ).compute_metrics()

    # # Evaluate the direct model
    # elif mode == 'direct-model':
    #     DirectModelMetricsComputation(
    #         checkpoint_path='/Users/jan/Downloads/unet3d2_epoch=18_val_loss=0.13242.ckpt',
    #         dataset_root='/Users/jan/Downloads/openneuro-datasets/preprocessed/ds*/',
    #         device='cpu'
    #     ).compute_metrics()

    # # Evaluate the SynBo-Disco baseline model
    # elif mode == 'baseline':
    #     BaselineModelMetricsComputation(
    #         subject_paths='/home/mlc/dev/fmdc/fmdc-ai-thesis/scripts/INPUTS/ds*/sub-*',
    #         device='cpu'
    #     ).compute_metrics()

    # # Evaluate the mean fieldmap model
    # elif mode == 'mean-fieldmaps':
    #     MeanFieldmapsMetricsComputation(
    #         subject_paths='/Users/jan/Downloads/openneuro-datasets/preprocessed-average-fieldmaps/ds*/sub-*/',
    #         device='cpu'
    #     ).compute_metrics()
