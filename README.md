<div align="center">    
 
# Field Map AI

</div>
 
## Description
Estimate an undistorted fMRI scan from a distorted scan and an accompanying T1-weighted scan.

## Project Structure
- The `project` directory contains the main part of the code. Top-level Python files can be executed directly.
- The `scripts` directory contains useful small standalone scripts for data preparation and model visualization.

## Running it
Install all dependencies in an environment management tool of your choice. This has been tested with conda.

### Training
The following command will train a model from scratch, given a dataset that has been preprocessed such that it can be used by this project. Training will be performed on the fieldmap-based model variant and will take around 15 hours on a TITAN V GPU. If you want to train the direct model instead you can just replace the underlying model in the source code.

```bash
python project/training.py --training_dataset_path=/path/to/preprocessed/datasets/ds*/ --checkpoint_path=/path/to/checkpoint/output/ --max_epochs=100000 --batch_size=32
```

For convenience (and while my storage space allows), you can also download pre-trained checkpoints of the model. The checkpoint that has been used for evaluation in my thesis can be found [here](https://drive.google.com/file/d/1KuMoE_z6MD-NTB9IU9OVSDh6DA9KmdZM/view?usp=sharing). A set of ten checkpoints created during k-fold validation can be downloaded [here](https://drive.google.com/file/d/1_T0NINnIIHtZHG17kQIVCLLV793EtZV3/view?usp=sharing).

### Inference
Given a single sample from a dataset and a pre-trained model, `project/inference.py` will run inference on that sample in order to generate the undistorted fMRI approximation.

### Evaluation
The main script for evaluating models can be found in `project/metrics_scripts/compute_metrics.py`. Depending on the mode you set it to, it can compute a set of metrics for the direct model, the fieldmap model, and both baseline models over a given validation/test dataset. For all samples, it outputs SSIM, MSE, and correlation coefficients, as well as aggregates of these.

Furthermore, the script in `project/infer_correlation_matrices.py` can be used to create matrices of pair-wise temporal correlation between the ground-truth undistorted images and the results from both the direct and the field map model. These are saved as Numpy arrays and as corresponding VTI files to be (volume-)visualized with Paraview.

### Miscellaneous Scripts
The `scripts` directory contains a bunch of useful scripts.
- `prepare_average_fieldmap_data.py` aggregates all fieldmaps from the training split to create the model for the mean fieldmap baseline.
- `prepare_training_data.py` does everything that is necessary to convert OpenNeuro datasets into preprocessed datasets to be used with this project.
- `run_baseline_model_on_validation_samples.py` runs the paper baseline model on all samples of the validation split. These outputs can in turn be used to compute evaluation metrics.
- `sync_wandb.py` repeatedly synchronizes wandb runs from the firewalled GPU server used for training by mounting its data volume.
- `visualize_models.py` creates graphs visualizing the model architectures.

## Next Steps
There are multiple open tasks that require further exploration:

- [x] Train and evaluate models for various dataset splits, allowing for a better quantification of model performance
- [ ] Enhance the dataset by adding more OpenNeuro datasets with varying subject health and scanner types (see `dataset-candidates.md` for a list of datasets that come into question)
- [x] Compute SSIM as a metric
- [ ] Re-align and distortion-correct data after inference to show general purpose
- [ ] For the temporal correlation metric, do a z-transform of the r-values and do voxel-wise t-tests
- [ ] Also show best/worst subjects as examples rather than random ones
- [ ] Add a scatterplot similar to figure 5.7, but for each model/baseline scatterplot the performance on all subjects allowing for inspection of their distribution shape, potentially alos add a convex function that fits an ellipsoid
