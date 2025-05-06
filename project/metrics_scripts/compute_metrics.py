# Import the dependencies
import glob
import argparse
import json
from sklearn.model_selection import KFold
from project.data import fmri_data_util
from project.metrics_scripts.synbo_disco_model import Synb0MetricsComputation
from project.metrics_scripts.direct_model import DirectModelMetricsComputation
from project.metrics_scripts.fieldmaps_model import FieldmapsModelMetricsComputation
from project.metrics_scripts.k_fold_fieldmaps_model import KFoldFieldmapsModelMetricsComputation
from project.metrics_scripts.mean_fieldmaps import MeanFieldmapsMetricsComputation
import os

"""
Compute all performance metrics on a model
"""
def main():
    # Parse and load arguments
    parser = argparse.ArgumentParser(description="Compute performance metrics for the fieldmap model.")
    parser.add_argument("--CHECKPOINT_PATH", help="Path to the model checkpoint.")
    parser.add_argument("--TEST_JSON_PATH", required=True, help="Path to the JSON file containing test paths.")
    parser.add_argument("--device", default="cpu", help="Device to run evaluation on (e.g., 'cpu' or 'cuda').")
    parser.add_argument("--model", choices=["fieldmap", "mean-fieldmaps", "kfcv", "kfcv-2", "synbold", "synb0", "kfcv-m0", "kfcv-m1", "kfcv-m2", "kfcv-m3", "kfcv-m4", "kfcv-m5", "kfcv-m6", "kfcv-m7", "kfcv-m8", "kfcv-m9"], default="fieldmap", help="Which model to run the evaluation on")
    args = parser.parse_args()
    # CHECKPOINT_PATH = args.CHECKPOINT_PATH
    # TEST_JSON_PATH = args.TEST_JSON_PATH
    # device = args.device

    # Load the test split from JSONow 
    with open(args.TEST_JSON_PATH, "r") as f:
        test_data = json.load(f)
    # TEST_PATHS=test_data["test_paths"][:2] # Uncomment for testing
    TEST_PATHS=test_data["test_paths"]
    print(f"Processing {len(TEST_PATHS)} test samples")

    # Test case for the fieldmap model
    if args.model=="fieldmap":
        metrics_comp = FieldmapsModelMetricsComputation(
            CHECKPOINT_PATH=args.CHECKPOINT_PATH,
            TEST_PATHS=TEST_PATHS,
            device=args.device
        )
        report_root = "/indirect/student/magnuschristensen/dev/fmdc/fmdc-ai-thesis/reports/fieldmap"
        report_file = os.path.join(report_root, "fieldmap_test_m0.txt")
        times_file = os.path.join(report_root, "fieldmap_times_test_m0.txt")
        metrics_comp.compute_metrics(report_file=report_file)
        metrics_comp.save_compute_times(save_path=times_file)

    # Test suite for the kfcv case
    elif args.model=="kfcv":
        # report_root = "/indirect/student/magnuschristensen/dev/fmdc/fmdc-ai-thesis/reports/kfcv"
        report_root = "/indirect/student/magnuschristensen/dev/fmdc/fmdc-ai-thesis/reports/kfcv/PA"
        model_root = "/indirect/student/magnuschristensen/dev/fmdc/downloads/kfcv-ckpt-AP"
        print(f"Report root: {report_root}")
        models = [
            "mkilsf32_model0_unet3d_epoch=147_val_loss=1399.99133.ckpt",
            "sc2avov0_model1_unet3d_epoch=209_val_loss=1305.05005.ckpt",
            "rfotmw47_model2_unet3d_epoch=234_val_loss=1317.46313.ckpt",
            "18dr2qm4_model3_unet3d_epoch=172_val_loss=1334.92322.ckpt",
            "t1z5jqsm_model4_unet3d_epoch=159_val_loss=1650.42480.ckpt",
            "ozsqm0hm_model5_unet3d_epoch=190_val_loss=1318.22595.ckpt",
            "hsgrha8b_model6_unet3d_epoch=198_val_loss=1163.12549.ckpt",
            "bnn21ltx_model7_unet3d_epoch=191_val_loss=1307.36890.ckpt",
            "844vqvn2_model8_unet3d_epoch=225_val_loss=1432.07471.ckpt",
            "zne5b47m_model9_unet3d_epoch=195_val_loss=1117.42761.ckpt"
        ]

        for idx, model in enumerate(models, start=1):
            print(f"At idx {idx}, Model: {model}\n")
            full_model_path = os.path.join(model_root, model)
            kfcv_report_file = os.path.join(report_root, f"model{idx}_metrics_report.txt")
            kfcv_times_file = os.path.join(report_root, f"model{idx}_compute_times.txt")
            metrics_comp = FieldmapsModelMetricsComputation(
                CHECKPOINT_PATH=full_model_path,
                TEST_PATHS=TEST_PATHS,
                device=args.device
            )
            metrics_comp.compute_metrics(report_file=kfcv_report_file)
            metrics_comp.save_compute_times(save_path=kfcv_times_file)

    
    # Test suite for the kfcv case
    elif args.model=="kfcv-2":
        # report_root = "/indirect/student/magnuschristensen/dev/fmdc/fmdc-ai-thesis/reports/kfcv"
        report_root = "/indirect/student/magnuschristensen/dev/fmdc/fmdc-ai-thesis/reports/kfcv/RL"
        model_root = "/indirect/student/magnuschristensen/dev/fmdc/downloads/kfcv-ckpt-AP"
        print(f"Report root: {report_root}")
        models = [
            "mkilsf32_model0_unet3d_epoch=147_val_loss=1399.99133.ckpt",
            "sc2avov0_model1_unet3d_epoch=209_val_loss=1305.05005.ckpt",
            "rfotmw47_model2_unet3d_epoch=234_val_loss=1317.46313.ckpt",
            "18dr2qm4_model3_unet3d_epoch=172_val_loss=1334.92322.ckpt",
            "t1z5jqsm_model4_unet3d_epoch=159_val_loss=1650.42480.ckpt",
            "ozsqm0hm_model5_unet3d_epoch=190_val_loss=1318.22595.ckpt",
            "hsgrha8b_model6_unet3d_epoch=198_val_loss=1163.12549.ckpt",
            "bnn21ltx_model7_unet3d_epoch=191_val_loss=1307.36890.ckpt",
            "844vqvn2_model8_unet3d_epoch=225_val_loss=1432.07471.ckpt",
            "zne5b47m_model9_unet3d_epoch=195_val_loss=1117.42761.ckpt"
        ]

        for idx, model in enumerate(models, start=1):
            print(f"At idx {idx}, Model: {model}\n")
            full_model_path = os.path.join(model_root, model)
            kfcv_report_file = os.path.join(report_root, f"model{idx}_metrics_report.txt")
            kfcv_times_file = os.path.join(report_root, f"model{idx}_compute_times.txt")
            metrics_comp = FieldmapsModelMetricsComputation(
                CHECKPOINT_PATH=full_model_path,
                TEST_PATHS=TEST_PATHS,
                device=args.device
            )
            metrics_comp.compute_metrics(report_file=kfcv_report_file)
            metrics_comp.save_compute_times(save_path=kfcv_times_file)

    # Baselines
    elif args.model == "synbold":
        pass

    elif args.model == "synb0":
        report_root = "/indirect/student/magnuschristensen/dev/fmdc/fmdc-ai-thesis/reports/synb0"
        synb0_report_file = os.path.join(report_root, "synb0_report.txt")
        # synb0_times_file = os.path.join(report_root, "synb0_times.txt")
        metrics_comp = Synb0MetricsComputation(
            subject_paths=TEST_PATHS,
            device=args.device
        )
        metrics_comp.compute_metrics(report_file=synb0_report_file)
        # metrics_comp.save_compute_times(save_path=synb0_times_file)

    elif args.model == "mean-fieldmaps":
        report_root = "/indirect/student/magnuschristensen/dev/fmdc/fmdc-ai-thesis/reports/mean_fieldmap"
        mean_fmap_report_file = os.path.join(report_root, "mean_fmap_report_AP.txt")
        mean_fmap_times_file = os.path.join(report_root, "mean_fmap_times_AP.txt")
        metrics_comp = MeanFieldmapsMetricsComputation(
            subject_paths=TEST_PATHS,
            device = args.device
        )
        metrics_comp.compute_metrics(report_file=mean_fmap_report_file)
        metrics_comp.save_compute_times(save_path=mean_fmap_times_file)
        pass

    elif args.model == "kfcv-m0":
        model_checkpoint = "/indirect/student/magnuschristensen/dev/fmdc/downloads/k-fold-checkpoints/6wvvsz8o_model0_unet3d_epoch=139_val_loss=1358.73145.ckpt"
        metrics_comp = FieldmapsModelMetricsComputation(
            CHECKPOINT_PATH=model_checkpoint,
            TEST_PATHS=TEST_PATHS,
            device=args.device
        )
        report_root = "/indirect/student/magnuschristensen/dev/fmdc/fmdc-ai-thesis/reports/kfcv"
        report_file = os.path.join(report_root, "model0_metrics_report.txt")
        times_file = os.path.join(report_root, "model0_compute_times.txt")
        metrics_comp.compute_metrics(report_file=report_file)
        metrics_comp.save_compute_times(save_path=times_file)

    elif args.model == "kfcv-m1":
        model_checkpoint = "/indirect/student/magnuschristensen/dev/fmdc/downloads/k-fold-checkpoints/j9vd5pfe_model1_unet3d_epoch=139_val_loss=1150.53113.ckpt"
        metrics_comp = FieldmapsModelMetricsComputation(
            CHECKPOINT_PATH=model_checkpoint,
            TEST_PATHS=TEST_PATHS,
            device=args.device
        )
        report_root = "/indirect/student/magnuschristensen/dev/fmdc/fmdc-ai-thesis/reports/kfcv"
        report_file = os.path.join(report_root, "model1_metrics_report.txt")
        metrics_comp.compute_metrics(report_file=report_file)

    elif args.model == "kfcv-m2":
        model_checkpoint = "/indirect/student/magnuschristensen/dev/fmdc/downloads/k-fold-checkpoints/qa4714fw_model2_unet3d_epoch=129_val_loss=1155.35828.ckpt"
        metrics_comp = FieldmapsModelMetricsComputation(
            CHECKPOINT_PATH=model_checkpoint,
            TEST_PATHS=TEST_PATHS,
            device=args.device
        )
        report_root = "/indirect/student/magnuschristensen/dev/fmdc/fmdc-ai-thesis/reports/kfcv"
        report_file = os.path.join(report_root, "model2_metrics_report.txt")
        metrics_comp.compute_metrics(report_file=report_file)


    elif args.model == "kfcv-m3":
        model_checkpoint = "/indirect/student/magnuschristensen/dev/fmdc/downloads/k-fold-checkpoints/z96lsd46_model3_unet3d_epoch=139_val_loss=1245.35901.ckpt"
        metrics_comp = FieldmapsModelMetricsComputation(
            CHECKPOINT_PATH=model_checkpoint,
            TEST_PATHS=TEST_PATHS,
            device=args.device
        )
        report_root = "/indirect/student/magnuschristensen/dev/fmdc/fmdc-ai-thesis/reports/kfcv"
        report_file = os.path.join(report_root, "model3_metrics_report.txt")
        metrics_comp.compute_metrics(report_file=report_file)

    elif args.model == "kfcv-m4":
        model_checkpoint = "/indirect/student/magnuschristensen/dev/fmdc/downloads/k-fold-checkpoints/54p2097p_model4_unet3d_epoch=129_val_loss=1330.44702.ckpt"
        metrics_comp = FieldmapsModelMetricsComputation(
            CHECKPOINT_PATH=model_checkpoint,
            TEST_PATHS=TEST_PATHS,
            device=args.device
        )
        report_root = "/indirect/student/magnuschristensen/dev/fmdc/fmdc-ai-thesis/reports/kfcv"
        report_file = os.path.join(report_root, "model4_metrics_report.txt")
        metrics_comp.compute_metrics(report_file=report_file)
        # metrics_comp.save_compute_times

    elif args.model == "kfcv-m5":
        model_checkpoint = "/indirect/student/magnuschristensen/dev/fmdc/downloads/k-fold-checkpoints/goputb7m_model5_unet3d_epoch=129_val_loss=1147.49390.ckpt"
        metrics_comp = FieldmapsModelMetricsComputation(
            CHECKPOINT_PATH=model_checkpoint,
            TEST_PATHS=TEST_PATHS,
            device=args.device
        )
        report_root = "/indirect/student/magnuschristensen/dev/fmdc/fmdc-ai-thesis/reports/kfcv"
        report_file = os.path.join(report_root, "model5_metrics_report.txt")
        metrics_comp.compute_metrics(report_file=report_file)

    elif args.model == "kfcv-m6":
        model_checkpoint = "/indirect/student/magnuschristensen/dev/fmdc/downloads/k-fold-checkpoints/81421cn0_model6_unet3d_epoch=119_val_loss=1355.11401.ckpt"
        metrics_comp = FieldmapsModelMetricsComputation(
            CHECKPOINT_PATH=model_checkpoint,
            TEST_PATHS=TEST_PATHS,
            device=args.device
        )
        report_root = "/indirect/student/magnuschristensen/dev/fmdc/fmdc-ai-thesis/reports/kfcv"
        report_file = os.path.join(report_root, "model6_metrics_report.txt")
        metrics_comp.compute_metrics(report_file=report_file)
        
    elif args.model == "kfcv-m7":
        model_checkpoint = "/indirect/student/magnuschristensen/dev/fmdc/downloads/k-fold-checkpoints/6qud2br2_model7_unet3d_epoch=139_val_loss=1509.06543.ckpt"
        metrics_comp = FieldmapsModelMetricsComputation(
            CHECKPOINT_PATH=model_checkpoint,
            TEST_PATHS=TEST_PATHS[:1],
            device=args.device
        )
        report_root = "/indirect/student/magnuschristensen/dev/fmdc/fmdc-ai-thesis/reports/kfcv"
        report_file = os.path.join(report_root, "model7_metrics_report.txt")
        metrics_comp.compute_metrics(report_file=report_file)
        
    elif args.model == "kfcv-m8":
        model_checkpoint = "/indirect/student/magnuschristensen/dev/fmdc/downloads/k-fold-checkpoints/486hlbe0_model8_unet3d_epoch=129_val_loss=1269.33875.ckpt"
        metrics_comp = FieldmapsModelMetricsComputation(
            CHECKPOINT_PATH=model_checkpoint,
            TEST_PATHS=TEST_PATHS,
            device=args.device
        )
        report_root = "/indirect/student/magnuschristensen/dev/fmdc/fmdc-ai-thesis/reports/kfcv"
        report_file = os.path.join(report_root, "model8_metrics_report.txt")
        metrics_comp.compute_metrics(report_file=report_file)

    elif args.model == "kfcv-m9":
        model_checkpoint = "/indirect/student/magnuschristensen/dev/fmdc/downloads/k-fold-checkpoints/ht2ldc2u_model9_unet3d_epoch=129_val_loss=1357.09900.ckpt"
        metrics_comp = FieldmapsModelMetricsComputation(
            CHECKPOINT_PATH=model_checkpoint,
            TEST_PATHS=TEST_PATHS,
            device=args.device
        )
        report_root = "/indirect/student/magnuschristensen/dev/fmdc/fmdc-ai-thesis/reports/kfcv"
        report_file = os.path.join(report_root, "model9_metrics_report.txt")
        metrics_comp.compute_metrics(report_file=report_file)

    # Graceful error handling
    else:
        raise ValueError("Unknown model type provided")
    
    
        # root = "/indirect/student/magnuschristensen/dev/fmdc/downloads/k-fold-checkpoints"

    # metrics_comp.compute_metrics(report_file=report_file)
    
    # elif model=="k-fold-cross-validation":
    #     dataset_paths = TEST_PATHS
    #     k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    #     model_paths = [
    #         "ejla3ly1_model0_unet3d2_epoch=129_val_loss=1357.99377_03-2025.ckpt",
    #         "zkvba6sq_model1_unet3d2_epoch=139_val_loss=1338.77039_04-2025.ckpt",
    #         "g8pmspp9_model2_unet3d2_epoch=139_val_loss=1294.42419_04-2025.ckpt",
    #         "xwp1h25f_model3_unet3d2_epoch=129_val_loss=1319.20850_05-2025.ckpt",
    #         "rkkpv136_model4_unet3d2_epoch=129_val_loss=1359.35706_05-2025.ckpt"
    #     ]
    #     full_model_paths = [os.path.join(CHECKPOINT_PATH, "k-fold_cross_validation", f"model_{idx}", model) for idx, model in enumerate(model_paths)]

    #     for fold, (_, val_idx) in enumerate(k_fold.split(dataset_paths)):
    #         test_paths = [dataset_paths[index] for index in val_idx]
    #         print(f"Fold {fold}:")
    #         print(test_paths)

    #         KFoldFieldmapsModelMetricsComputation(
    #             CHECKPOINT_PATH=full_model_paths[fold],
    #             TEST_PATHS=test_paths,
    #             device=device
    #         ).compute_metrics()
        




            # KFoldFieldmapsModelMetricsComputation(
            #     checkpoint_path=full_model_paths[fold],
            #     subject_paths=test_paths
            # )

if __name__ == '__main__':
    main()


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


    # ---------------------------------------------------------


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