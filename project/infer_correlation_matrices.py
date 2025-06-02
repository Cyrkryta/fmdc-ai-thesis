import os.path
import tempfile
from glob import glob
from os.path import abspath

import nibabel as nib
import nipype.interfaces.io as nio
import numpy as np
import torch
import volumentations.augmentations.functional as F
from nipype import Node, SelectFiles, Workflow
from nipype.interfaces import fsl
from tqdm import tqdm

import vtk
import vtk.util.numpy_support as numpy_support

from data import fmri_data_util, data_util
from models.unet3d_fieldmap import UNet3DFieldmap
from project.metrics.metrics import TemporalCorrelation
from project.models.unet3d_direct import UNet3DDirect


def _load_input_sample_affines(input_subject_path):
    t1_path = os.path.join(input_subject_path, 'T1w.nii.gz')
    b0_d_path = os.path.join(input_subject_path, 'b0_d.nii.gz')
    b0_u_path = os.path.join(input_subject_path, 'b0_u.nii.gz')
    mask_path = os.path.join(input_subject_path, 'b0_mask.nii.gz')
    fieldmap_path = os.path.join(input_subject_path, 'field_map.nii.gz')

    return nib.load(t1_path).affine, nib.load(b0_d_path).affine, nib.load(b0_u_path).affine, nib.load(mask_path).affine, nib.load(fieldmap_path).affine


def get_subject_paths(dataset_pattern):
    subject_paths = fmri_data_util.collect_all_subject_paths(dataset_paths=glob(dataset_pattern))

    total_count = len(subject_paths)
    train_count = int(0.7 * total_count)
    val_count = int(0.2 * total_count)
    test_count = total_count - train_count - val_count

    rng = torch.Generator()
    rng.manual_seed(0)
    _, val_paths, _ = torch.utils.data.random_split(
        subject_paths, (train_count, val_count, test_count),
        generator=rng
    )

    return list(val_paths)


def warp_to_mni(input_volume, affine_input_volume, subject_path):
    with tempfile.TemporaryDirectory() as directory:
        input_volume = np.transpose(input_volume, axes=(1, 2, 0))
        input_volume_image = nib.Nifti1Image(input_volume, affine_input_volume)
        nib.save(input_volume_image, os.path.join(directory, 'input.nii.gz'))

        in_volume = Node(SelectFiles({"out_file": abspath(os.path.join(directory, 'input.nii.gz'))}), name="in_volume")
        in_warp = Node(SelectFiles({"out_file": abspath(os.path.join(subject_path, 'T1w_to_MNI_warp.nii.gz'))}), name="in_warp")
        in_premat = Node(SelectFiles({"out_file": abspath(os.path.join(subject_path, 'func2struct.mat'))}), name="in_premat")
        out_volume = Node(nio.ExportFile(out_file=abspath(os.path.join(directory, "output.nii.gz")), clobber=True), name="out_volume")

        ref = "/home/mlc/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz"
        
        apply_warp = Node(fsl.preprocess.ApplyWarp(ref_file=ref), name="apply_warp")

        workflow = Workflow(name="warp_to_mni")

        workflow.connect(in_volume, "out_file", apply_warp, "in_file")
        workflow.connect(in_warp, "out_file", apply_warp, "field_file")
        workflow.connect(in_premat, "out_file", apply_warp, "premat")
        workflow.connect(apply_warp, "out_file", out_volume, "in_file")

        workflow.run()

        out = nib.load(os.path.join(directory, 'output.nii.gz'))
        fdata = out.get_fdata()
        affine = out.affine

    return fdata, affine


def _debug_save_nifti(volume, affine, name):
    img = nib.Nifti1Image(volume, affine)
    nib.save(img, os.path.join('/home/mlc/dev/fmdc/downloads/correlation-matrices/nifti', f'{name}.nii.gz'))


def load_input_samples(subject_path):
    img_t1_all, img_b0_d_all, img_b0_u_all, img_mask_all, img_fieldmap_all, b0u_affine_all, fieldmap_affine_all, echo_spacing_all = fmri_data_util.load_data_from_path(subject_path)
    affine_t1, affine_b0_d, affine_b0_u, affine_mask, affine_fieldmap = _load_input_sample_affines(subject_path)
    time_series = []

    for time_step in range(len(list(img_t1_all))):
        img_t1 = list(img_t1_all)[time_step]
        img_b0_d = list(img_b0_d_all)[time_step]
        img_b0_u = list(img_b0_u_all)[time_step]
        img_mask = list(img_mask_all)[time_step]

        img_warped_t1, affine_warped_t1 = warp_to_mni(img_t1, affine_t1, subject_path)
        img_warped_b0_d, affine_warped_b0_d = warp_to_mni(img_b0_d, affine_b0_d, subject_path)
        img_warped_b0_u, affine_warped_b0_u = warp_to_mni(img_b0_u[0], affine_b0_u, subject_path)
        img_warped_mask, affine_warped_mask = warp_to_mni(img_mask[0], affine_mask, subject_path)

        img_warped_b0_d[img_warped_mask == 0] = -1
        img_warped_b0_u[img_warped_mask == 0] = -1

        shape = (64, 64, 36)
        img_warped_t1 = F.resize(img_warped_t1, new_shape=shape, interpolation=0)
        img_warped_b0_d = F.resize(img_warped_b0_d, new_shape=shape, interpolation=0)
        img_warped_b0_u = F.resize(img_warped_b0_u, new_shape=shape, interpolation=0)
        img_warped_mask = F.resize(img_warped_mask, new_shape=shape, interpolation=0)

        img_warped_t1 = np.transpose(img_warped_t1, axes=(2, 0, 1))
        img_warped_b0_d = np.transpose(img_warped_b0_d, axes=(2, 0, 1))
        img_warped_b0_u = np.transpose(img_warped_b0_u, axes=(2, 0, 1))
        img_warped_mask = np.transpose(img_warped_mask, axes=(2, 0, 1))

        img_warped_b0_u = np.expand_dims(img_warped_b0_u, axis=0)
        img_warped_mask = np.expand_dims(img_warped_mask, axis=0)

        img_data = np.stack((img_warped_b0_d, img_warped_t1))
        b0u_affine = list(b0u_affine_all)[time_step]
        fieldmap_affine = list(fieldmap_affine_all)[time_step]
        echo_spacing = list(echo_spacing_all)[time_step]

        time_series.append({
            'img': img_data,
            't1': img_warped_t1,
            'b0d': img_warped_b0_d,
            'b0u': img_warped_b0_u,
            'mask': img_warped_mask,
            'b0u_affine': b0u_affine,
            'fieldmap_affine': fieldmap_affine,
            'echo_spacing': echo_spacing
        })

    return time_series


def _undistort_direct(model, sample, device):
    input_img = torch.as_tensor(sample['img']).float().to(device)
    out = model(input_img.unsqueeze(0))
    return out.squeeze().cpu().detach().numpy()


def _undistort_fieldmap(model, sample, device):
    input_img = torch.as_tensor(sample['img']).float().to(device)
    input_b0d = torch.as_tensor(sample['b0d']).float().to(device)
    input_b0u_affine = torch.as_tensor(sample['b0u_affine']).float().to(device)
    input_fieldmap_affine = torch.as_tensor(sample['fieldmap_affine']).float().to(device)
    input_echo_spacing = torch.as_tensor(sample['echo_spacing']).float().to(device)
    out = model(input_img.unsqueeze(0))
    result = model._undistort_b0(input_b0d, out[0], input_b0u_affine, input_fieldmap_affine, input_echo_spacing)
    return data_util.nii2torch(result)[0]


if __name__ == '__main__':
    """
    Create matrices of pair-wise temporal correlation between the ground-truth undistorted images and the results from 
    both the direct and the field map model. Each entry in these matrices are Pearson coefficients, which are then 
    stored as both Numpy files as well as VTI files to be visualized in Paraview.
    """

    subject_paths = get_subject_paths('/home/mlc/dev/fmdc/downloads/openneuro-datasets/preprocessed/ds*/')
    output_path = '/home/mlc/dev/fmdc/downloads/correlation-matrices/v2/'
    fieldmap_ckpt_path = '/home/mlc/dev/fmdc/downloads/fmri-checkpoints/last.ckpt'
    device = 'cpu'

    # direct_model = UNet3DDirect.load_from_checkpoint('/Users/jan/Downloads/unet3d2_epoch=18_val_loss=0.13242.ckpt', map_location=torch.device(device), encoder_map_location=torch.device(device), device=device)
    # direct_model.to(device)
    # direct_model.eval()
    fieldmap_model = UNet3DFieldmap.load_from_checkpoint(fieldmap_ckpt_path, map_location=torch.device(device), encoder_map_location=torch.device(device), device=device)
    fieldmap_model.to(device)
    fieldmap_model.eval()

    for subject_path in tqdm(subject_paths):
        subject_output_path = os.path.join(output_path, subject_path.split('/')[-2], subject_path.split('/')[-1])
        if os.path.isdir(subject_output_path):
            continue

        # temporal_correlation_out_direct = TemporalCorrelation()
        temporal_correlation_out_fieldmap = TemporalCorrelation()
        temporal_correlation_distorted = TemporalCorrelation()

        # correlations_direct_out = []
        # correlations_direct_distorted = []
        # correlations_direct_delta = []
        correlations_fieldmap_out = []
        correlations_fieldmap_distorted = []
        correlations_fieldmap_delta = []

        for sample in load_input_samples(subject_path):
            # direct_out = _undistort_direct(direct_model, sample, device)
            # temporal_correlation_out_direct.update(
            #     ground_truth=sample['b0u'].squeeze(),
            #     image=np.where(sample['mask'], direct_out, -1).squeeze()
            # )

            fieldmap_out = _undistort_fieldmap(fieldmap_model, sample, device)
            temporal_correlation_out_fieldmap.update(
                ground_truth=sample['b0u'].squeeze(),
                image=np.where(sample['mask'], fieldmap_out, -1).squeeze()
            )

            temporal_correlation_distorted.update(
                ground_truth=sample['b0u'].squeeze(),
                image=sample['b0d']
            )

        # pearson_coefficients_out_direct, _ = temporal_correlation_out_direct.compute()
        pearson_coefficients_out_fieldmap, _ = temporal_correlation_out_fieldmap.compute()
        pearson_coefficients_distorted, _ = temporal_correlation_distorted.compute()

        if not os.path.exists(subject_output_path):
            os.makedirs(subject_output_path)

        # with open(f'{subject_output_path}/pearson_coefficients_out_direct.npy', 'wb+') as f:
        #     np.save(f, pearson_coefficients_out_direct)
        with open(f'{subject_output_path}/pearson_coefficients_out_fieldmap.npy', 'wb+') as f:
            np.save(f, pearson_coefficients_out_fieldmap)
        with open(f'{subject_output_path}/pearson_coefficients_distorted.npy', 'wb+') as f:
            np.save(f, pearson_coefficients_distorted)
        # with open(f'{subject_output_path}/pearson_coefficients_delta_direct.npy', 'wb+') as f:
        #     np.save(f, pearson_coefficients_distorted - pearson_coefficients_out_direct)
        with open(f'{subject_output_path}/pearson_coefficients_delta_fieldmap.npy', 'wb+') as f:
            np.save(f, pearson_coefficients_distorted - pearson_coefficients_out_fieldmap)

    aggregates = {
        "pearson_coefficients_distorted": [],
        # "pearson_coefficients_out_direct": [],
        "pearson_coefficients_out_fieldmap": [],
        # "pearson_coefficients_delta_direct": [],
        "pearson_coefficients_delta_fieldmap": []
    }

    for subject_path in tqdm(subject_paths):
        subject_output_path = os.path.join(output_path, subject_path.split('/')[-2], subject_path.split('/')[-1])

        for aggregate_name in aggregates.keys():
            aggregates[aggregate_name].append(np.load(os.path.join(subject_output_path, f'{aggregate_name}.npy')))

    for aggregate_name, aggregate_values in aggregates.items():
        aggregate_mean = np.nanmean(np.array(aggregate_values), axis=0)

        data_type = vtk.VTK_FLOAT
        shape = aggregate_mean.shape

        flat_data_array = aggregate_mean.flatten()
        vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)

        img = vtk.vtkImageData()
        img.GetPointData().SetScalars(vtk_data)
        img.SetDimensions(shape[2], shape[1], shape[0])

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(os.path.join(output_path, f'mean_{aggregate_name}.vti'))
        writer.SetInputData(img)
        writer.Write()
