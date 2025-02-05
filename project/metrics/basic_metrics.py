import numpy as np
import nibabel as nib

def compare_nifti_files(f1: str, f2: str, gt: str):
    data_f1 = nib.load(f1).get_fdata()
    data_f2 = nib.load(f2).get_fdata()
    data_gt = nib.load(gt).get_fdata()

    print(f"Distorted data shape: {data_f1.shape}")
    print(f"Undistorted data shape: {data_f2.shape}")
    print(f"Ground truth data shape: {data_gt.shape}")

    # print(data_f1)
    # print(data_f2)

    if data_f1.shape != data_f2.shape:
        return False
    
    return np.array_equal(data_f1, data_f2)

f1 = '/home/mlc/dev/fmdc/downloads/fmri-checkpoints/inf-ckpt-trained/b0_distorted.nii.gz'
f2 = '/home/mlc/dev/fmdc/downloads/fmri-checkpoints/inf-ckpt-trained/b0_undistorted.nii.gz'
gt = '/home/mlc/dev/fmdc/downloads/fmri-checkpoints/inf-ckpt-trained/b0_gt.nii.gz'

# Comparing the files
equality = compare_nifti_files(f1=f1, f2=f2, gt=gt)
if equality:
    print('The files are equal!')
else:
    print('The files are NOT equal!')