#!/usr/bin/env python3
from src.preprocessing import preprocess_input_data
from src.data_util import normalize_img, nii2torch
from src.model.unet3d_fieldmap import UNet3DFieldmap
import nibabel as nib
import numpy as np
import torch
import os

def check_files(bold, t1w):
    if not os.path.isfile(bold):
        raise FileNotFoundError(f"BOLD file {bold} not found.")
    if not os.path.isfile(t1w):
        raise FileNotFoundError(f"T1w file {t1w} not found.")
    
def prepare_model_tensor(outdir:str, device="cpu"):
    # Create the file paths
    bold_path = os.path.join(outdir, "bold_mc.nii.gz")
    t1w_path = os.path.join(outdir, "t1w_in_bold.nii.gz")

    # Error handling for file existence
    if not os.path.isfile(bold_path):
        raise FileNotFoundError(f"BOLD motion corrected file {bold_path} not found.")
    if not os.path.isfile(t1w_path):
        raise FileNotFoundError(f"T1w in BOLD file {t1w_path} not found")
    
    # Load the images data
    bold_img = nib.load(bold_path).get_fdata()
    t1w_img = nib.load(t1w_path).get_fdata()

    # Calculate the number of timesteps
    num_timesteps = bold_img.shape[3] if bold_img.ndim == 4 else 1
    if len(t1w_img.shape) == 3:
            t1w_img = np.repeat(t1w_img[None, :], num_timesteps, axis=0)
            t1w_img = np.transpose(t1w_img, axes=(1, 2, 3, 0))
    
    # Convert to torch tensors
    bold_img = nii2torch(bold_img)
    t1w_img = nii2torch(t1w_img)

    # Data normalization
    t1w_img = normalize_img(t1w_img, 150, 0, 1, -1)
    max_bold_img = np.percentile(bold_img, 99)
    min_bold_img = 0
    bold_img = normalize_img(bold_img, max_bold_img, min_bold_img, 1, -1)
    bold_img_affine = nib.load(bold_path).affine

    # Transposing the images
    bold_4d = np.transpose(bold_img, axes=(1, 2, 3, 0))
    t1w_4d = np.transpose(t1w_img, axes=(1, 2, 3, 0))
    mid_idx = bold_4d.shape[3] // 2
    bold_ref = bold_4d[:, :, :, mid_idx]
    t1w_ref = t1w_4d[:, :, :, mid_idx]
    input_img = np.stack((bold_ref, t1w_ref))
    print(f"Input image shape: {input_img.shape}")
    input_tensor = torch.from_numpy(input_img).unsqueeze(0).float().to(device)
    print(f"Input tensor shape: {input_tensor.shape}")
    return input_tensor, bold_img_affine

# Predict fieldmap
def predict_fieldmap(input_tensor, ckpt_path, device="cpu"):
    model = UNet3DFieldmap.load_from_checkpoint(
        ckpt_path,
        map_location=torch.device(device),
        encoder_map_location=torch.device(device),
        device=device
    )
    model.to(device)
    model.eval()
    print(f"Estimating fieldmap...")
    with torch.no_grad():
        fmap_pred = model(input_tensor)[0].detach().cpu().numpy()
    print(f"Pred type: {type(fmap_pred)}, dimension: {fmap_pred.shape}")
    fmap_pred = fmap_pred[0]
    fmap_nifti = np.transpose(fmap_pred, (1, 2, 0))
    print(f"Final fieldmap shape: {fmap_nifti.shape}")
    return fmap_nifti

# Main script to run the preprocessing pipeline
def main():
    bold   = "/data/input/BOLD.nii.gz"
    t1w    = "/data/input/T1w.nii.gz"
    outdir = "/data/output"
    fieldmap_out = os.path.join(outdir, "fieldmap.nii.gz")
    ckpt_path = "/app/src/model/pt8df2wq_model0_unet3d_epoch=168_val_loss=1260.82800.ckpt"

    # Check if input files exist
    check_files(bold, t1w)

    # Call the preprocessing function
    res = preprocess_input_data(bold, t1w, outdir)
    bold_mc = res["bold_mc"]
    t1w_in_bold = res["t1w_in_bold"]
    print("Preprocessing complete:")
    print(f"bold_mc:     {bold_mc}")
    print(f"t1w_in_bold: {t1w_in_bold}")

    # Prepare the model tensor
    input_tensor, bold_img_affine = prepare_model_tensor(outdir)
    fieldmap = predict_fieldmap(input_tensor, ckpt_path)
    os.makedirs(outdir, exist_ok=True)
    nib.save(nib.Nifti1Image(fieldmap, bold_img_affine), fieldmap_out)
    print(f"Saved fieldmap to {fieldmap_out}")

    # Remove intermediate files
    for fn in (bold_mc, t1w_in_bold):
        os.remove(fn)

if __name__=="__main__":
    main()
