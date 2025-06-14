import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import scipy.ndimage
from nilearn.image import resample_img

def get_nifti_image(obj, obj_affine):
    nifti_img = nib.Nifti1Image(obj.detach().cpu().numpy(), obj_affine)
    return nifti_img  

def resample_image(source_nifti, target_affine, target_shape):
    # Define the source and target
    source_img = source_nifti
    target_params = (target_shape, target_affine)
    # Perform the resampling
    resampled_img = resample_from_to(source_img, target_params, order=1, mode="constant", cval=-100)
    # resampled_image = resample_img(source_nifti, target_affine, target_shape, interpolation, force_resample=True, copy_header=True)
    return resampled_img

def normalize_img(img, max_img, min_img, max, min):
    # Scale between [1 0]
    img = (img - min_img) / (max_img - min_img)

    # Scale between [max min]
    img = img * (max - min) + min

    return img


def unnormalize_img(img, max_img, min_img, max, min):
    # Undoes normalize_img()
    img = (img - min) / (max - min) * (max_img - min_img) + min_img

    return img

def load_only_nii(path_nii):
    nii = nib.load(path_nii)
    return nii


def get_nii_img(path_nii):
    nii = nib.load(path_nii)
    nii_img = nii.get_fdata()
    return nii_img


def nii2torch(nii_img):
    torch_img = np.transpose(nii_img, axes=(3, 2, 0, 1))
    return torch_img


def niiu2torch(nii_img):
    torch_img = np.expand_dims(nii_img, axis=0)
    torch_img = np.transpose(torch_img, axes=(4, 0, 3, 1, 2))
    return torch_img


def niimask2torch(nii_img, repetitions):
    torch_img = np.expand_dims(nii_img, axis=0)
    torch_img = np.repeat(torch_img[None, :], repetitions, axis=0)
    return np.transpose(torch_img, axes=(0, 1, 4, 2, 3))


def torch2nii(torch_img):
    nii_img = torch_img[0, :, :, :, :]
    nii_img = np.transpose(nii_img, axes=(2, 3, 1, 0))
    return nii_img


def random_unit_vector():
    theta = np.random.uniform(0, 2*np.pi)
    z = np.random.uniform(-1, 1)

    x = np.sqrt(1 - z ** 2)*np.cos(theta)
    y = np.sqrt(1 - z ** 2)*np.sin(theta)
    z = z

    return np.array([x, y, z])


def rodrigues2R(k, theta):
    # Get cross product matrix
    K = np.array([[    0, -k[2],  k[1]],
                  [ k[2],     0, -k[0]],
                  [-k[1],  k[0],    0]])

    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.matmul(K, K)


def Rt2xform(R, t):
    # concat R and t
    Rt = np.concatenate((R, t), axis=1)

    # concate [0, 0, 0, 1] to Rt to form affine matrix
    return np.concatenate((Rt, np.array([[0, 0, 0, 1]])), axis=0)


def apply_xform_vol(xform, vol):
    # Get voxel coordinates
    coords = np.meshgrid(np.arange(vol.shape[1]),
                         np.arange(vol.shape[0]),
                         np.arange(vol.shape[2]))

    xyz = np.vstack([coords[0].reshape(-1)-float(vol.shape[1]-1)/2,
                     coords[1].reshape(-1)-float(vol.shape[0]-1)/2,
                     coords[2].reshape(-1)-float(vol.shape[2]-1)/2,
                     np.ones(vol.shape).reshape(-1)])

    xyz_xform = np.matmul(xform, xyz)

    x = xyz_xform[0, :] + float(vol.shape[1]-1)/2
    y = xyz_xform[1, :] + float(vol.shape[0]-1)/2
    z = xyz_xform[2, :] + float(vol.shape[2]-1)/2

    x = x.reshape(vol.shape)
    y = y.reshape(vol.shape)
    z = z.reshape(vol.shape)

    return scipy.ndimage.map_coordinates(vol, [y, x, z], order=3)
