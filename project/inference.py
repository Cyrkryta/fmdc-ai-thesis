import os.path
import fnmatch

import nibabel as nib
import numpy as np
import torch

from data import fmri_data_util
from models.unet3d_fieldmap import UNet3DFieldmap

def find_subject_directories(root_dir, dir_name):
    matching_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in fnmatch.filter(dirnames, dir_name):
            matching_dirs.append(os.path.join(dirpath, dirname))
    return matching_dirs


def _torch_to_nii(img):
    img = np.expand_dims(img.cpu().detach().numpy(), axis=0)
    return np.transpose(img, axes=(2, 3, 1, 0))


def _store_data(t1, affine_t1, b0_distorted, affine_b0_d, b0_undistorted, b0_gt, affine_b0_u, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    s_b0_distorted = nib.Nifti1Image(_torch_to_nii(b0_distorted), affine=affine_b0_d)
    s_t1 = nib.Nifti1Image(_torch_to_nii(t1), affine=affine_t1)
    s_b0_undistorted = nib.Nifti1Image(b0_undistorted, affine=affine_b0_u)
    s_b0_gt = nib.Nifti1Image(_torch_to_nii(b0_gt), affine=affine_b0_u)

    nib.save(s_b0_distorted, os.path.join(output_path, 'b0_distorted.nii.gz'))
    nib.save(s_t1, os.path.join(output_path, 'T1.nii.gz'))
    nib.save(s_b0_undistorted, os.path.join(output_path, 'b0_undistorted.nii.gz'))
    nib.save(s_b0_gt, os.path.join(output_path, 'b0_gt.nii.gz'))


def _load_input_sample(input_subject_path, device):
    img_t1, img_b0_d, img_b0_u, img_mask, img_fieldmap, affine, _, echo_spacing = fmri_data_util.load_data_from_path(input_subject_path)
    img_t1 = torch.from_numpy(list(img_t1)[0]).float().to(device)
    img_b0_d = torch.from_numpy(list(img_b0_d)[0]).float().to(device)
    img_b0_u = torch.from_numpy(list(img_b0_u)[0]).float().to(device)
    img_mask = torch.from_numpy(list(img_mask)[0]).bool().to(device)
    img_fieldmap = torch.from_numpy(list(img_fieldmap)[0]).float().to(device)
    img_data = torch.stack((img_b0_d, img_t1))
    echo_spacing = torch.from_numpy(np.array(list(echo_spacing)[0])).float().to(device)
    return img_data, img_t1, img_b0_d, img_b0_u, img_mask, img_fieldmap, affine, echo_spacing


def _load_input_sample_affines(input_subject_path):
    t1_path = os.path.join(input_subject_path, 'T1w.nii.gz')
    b0_d_path = os.path.join(input_subject_path, 'b0_d.nii.gz')
    b0_u_path = os.path.join(input_subject_path, 'b0_u.nii.gz')
    mask_path = os.path.join(input_subject_path, 'b0_mask.nii.gz')
    fieldmap_path = os.path.join(input_subject_path, 'field_map.nii.gz')

    return nib.load(t1_path).affine, nib.load(b0_d_path).affine, nib.load(b0_u_path).affine, nib.load(mask_path).affine, nib.load(fieldmap_path).affine


def infer_and_store(input_subject_path, checkpoint_path, output_path, device):
    model = UNet3DFieldmap.load_from_checkpoint(checkpoint_path, map_location=torch.device(device), encoder_map_location=torch.device(device), device=device)
    model.to(device)
    model.eval()

    img_data, img_t1, img_b0_d, img_b0_u, img_mask, img_fieldmap, affine, echo_spacing = _load_input_sample(input_subject_path, device)
    affine_t1, affine_b0_d, affine_b0_u, affine_mask, affine_fieldmap = _load_input_sample_affines(input_subject_path)

    out = model(img_data.unsqueeze(0))

    b0_u = model._undistort_b0(img_b0_d, out[0], affine_b0_d, affine_fieldmap, echo_spacing)  # out[0]

    _store_data(
        t1=img_t1,
        affine_t1=affine_t1,
        b0_distorted=img_b0_d,
        affine_b0_d=affine_b0_d,
        b0_undistorted=b0_u,
        b0_gt=img_b0_u.squeeze(),
        affine_b0_u=affine_b0_u,
        output_path=output_path
    )

if __name__ == '__main__':
    subject_root_dir = '/home/mlc/dev/fmdc/downloads/openneuro-datasets/preprocessed/ds000224/'
    search_dir = "sub-*"
    matching_dirs = find_subject_directories(subject_root_dir, search_dir)
    infer_and_store(
        input_subject_path='/home/mlc/dev/fmdc/downloads/openneuro-datasets/preprocessed/ds000224/sub-MSC06', # My local input subject path
        output_path='/home/mlc/dev/fmdc/downloads/fmri-checkpoints/inf-ckpt-trained', # My local output path
        checkpoint_path='/home/mlc/dev/fmdc/downloads/fmri-checkpoints/last.ckpt', # My local checkpoint path
        device='cpu'
    )