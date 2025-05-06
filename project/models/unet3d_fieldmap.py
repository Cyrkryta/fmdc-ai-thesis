# Importing all of the dependencies
import logging
import os
import tempfile
from os.path import abspath
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from torch import optim
import nipype.interfaces.io as nio
from nipype.interfaces import fsl
from nipype import SelectFiles, Node, Function, Workflow
import nibabel as nib

from torchmetrics.functional import structural_similarity_index_measure as ssim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.losses import SSIMLoss
import subprocess


"""
Function for getting the median timeframe
"""
def GetMedianTF(in_file):
    import nibabel as nib
    return int(nib.load(in_file).header['dim'][4] / 2)

"""
Function for subtracting 5 from the input value
"""
def SubtractFive(in_value):
    return in_value - 5

"""
Class:
A 3D U-Net trained to predict a BOLD fMRI fieldmap
"""
class UNet3DFieldmap(pl.LightningModule):
    # Initialize the mode
    def __init__(self):
        super().__init__()
        self.model = UNet3D_2Module(2, 1)
        # self.ssim_weight = 0.1

    # Forward layer of the mode
    def forward(self, img):
        return self.model(img)

    # Training step
    def training_step(self, batch, batch_idx):
        img_data = batch["img_data"]
        fieldmap = batch["fieldmap"]
        out = self(img_data)
        train_loss = self.compute_loss(out, fieldmap)
        self.log("train_loss", train_loss, on_step=False, on_epoch=True)
        return train_loss


    # Validation step
    def validation_step(self, batch, batch_idx):
        img_data = batch["img_data"]
        fieldmap = batch["fieldmap"]
        mask = batch["mask"]
        out = self(img_data)
        val_loss = self.compute_loss(out, fieldmap)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        if batch_idx == 0:
            self._log_images(out, img_data, fieldmap, mask)
        return val_loss
    
    def undistort_full_sequence(self, BOLD_4d, T1_4D, affine_b0d, affine_fieldmap, echo_spacing, unwarp_direction, out_path=None):
        print(f"Preparing data for fieldmap retrieval...")
        b0_arr = BOLD_4d.cpu().detach().numpy() if torch.is_tensor(BOLD_4d) else BOLD_4d
        t1_arr = T1_4D.cpu().detach().numpy()   if torch.is_tensor(T1_4D) else T1_4D
        D, H, W, T = b0_arr.shape
        mid = T // 2
        bold_ref = b0_arr[..., mid]   # (D,H,W)
        t1_ref   = t1_arr[..., mid]   # (D,H,W)
        inp = np.stack([bold_ref, t1_ref]) # (2, D, H, W)
        inp_t = torch.from_numpy(inp).unsqueeze(0).float().to(self.device)
        print(f"Retrieving fieldmap...")
        with torch.no_grad():
            fmap_pred = self(inp_t)[0,0].cpu().detach().numpy() # (D, H, W)
        print(f"Fieldmap retrieved!")
        b0_for_fugue = np.transpose(b0_arr, (1, 2, 0, 3)) # (H, W, D, T)
        fmap_for_fugue = np.transpose(fmap_pred, (1, 2, 0)) # (H, W, D)

        print(f"Creating temporary folder...")
        with tempfile.TemporaryDirectory() as tmpdir:
            b0_path   = os.path.join(tmpdir, 'BOLD_4d.nii.gz')
            fmap_path = os.path.join(tmpdir, 'fmap_pred.nii.gz')
            nib.save(nib.Nifti1Image(b0_for_fugue, affine_b0d[0]),    b0_path)
            nib.save(nib.Nifti1Image(fmap_for_fugue, affine_b0d[0]),  fmap_path)
            if out_path is None:
                out_path = os.path.join(tmpdir, 'BOLD_undistorted.nii.gz')
            print(f"Running FUGUE....")
            cmd = [
                'fugue',
                '-i', b0_path,
                f'--loadfmap={fmap_path}',
                f'--dwell={echo_spacing}',
                '--smooth3=3',
                f'--unwarpdir={unwarp_direction}',
                '-u', out_path
            ]
            subprocess.run(cmd, check=True)
            und = nib.load(out_path).get_fdata() # (H, W, D, T)
            print(f"Returning undistorted 4D BOLD of shape: {und.shape}")
            return und

    # Logging images
    def _log_images(self, out, img, fieldmap, mask):
        # Pick a smple and slice
        sample_idx = 0
        slice_idx = img[sample_idx].shape[1] // 2

        # Retriieve the sampled images
        t1 = img[sample_idx][1][slice_idx]
        b0_d = img[sample_idx][0][slice_idx]
        fieldmap = fieldmap[sample_idx][0][slice_idx]
        mask = mask[sample_idx][0][slice_idx]

        # Log the sample images to wandb
        wandb.log({
            'epoch': self.current_epoch,
            't1': wandb.Image(t1, caption="T1w"),
            'b0_d': wandb.Image(b0_d, caption="B0 Distorted"),
            'fieldmap': wandb.Image(fieldmap, caption="Ground Truth Fieldmap"),
            'out': wandb.Image(out[sample_idx][0][slice_idx], caption="Model Output Fieldmap"),
            'mask': wandb.Image(mask, caption="B0 Mask")
        })

    # Function for defining and computing the loss function
    def compute_loss(self, out, fieldmap, mask=None):
        loss = F.mse_loss(out, fieldmap)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5)
        return optimizer
    

"""
Class:
Definition of 3D convolutional block
"""
class conv3D_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        # Call super
        super(conv3D_block, self).__init__()

        # Pytorch sequential layer containg conv layer, batch normalization and relu
        self.conv3D = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    # Create the forward layer
    def forward(self, x):
        x = self.conv3D(x)
        return x

class up_conv3D_block(nn.Module):

    def __init__(self, in_ch, out_ch, scale_tuple):

        super(up_conv3D_block, self).__init__()
        self.default_scale = scale_tuple

        self.up_conv3D = nn.Sequential(
            # nn.Upsample(scale_factor=scale_tuple, mode='trilinear'),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True), # increasing the depth by adding one below
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, target_size=None):
        """
        Forward method altered to interpolate
        """
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=self.default_scale, mode="trilinear", align_corners=False)
        x = self.up_conv3D(x)
        return x


class UNet3D_2Module(nn.Module):
    def __init__(self, n_in, n_out):
        super(UNet3D_2Module, self).__init__()

        filters_3D = [16, 16 * 2, 16 * 4, 16 * 8, 16 * 16, 16 * 16]  # = [16, 32, 64, 128, 256, 512]

        self.Conv3D_1 = conv3D_block(n_in, filters_3D[0])
        self.Conv3D_2 = conv3D_block(filters_3D[0], filters_3D[1])
        self.Conv3D_3 = conv3D_block(filters_3D[1], filters_3D[2])
        self.Conv3D_4 = conv3D_block(filters_3D[2], filters_3D[3])
        self.Conv3D_5 = conv3D_block(filters_3D[3], filters_3D[4])
        self.Conv3D_6 = conv3D_block(filters_3D[4], filters_3D[5])

        self.MaxPool3D_1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.MaxPool3D_2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.MaxPool3D_3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.MaxPool3D_4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.MaxPool3D_5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.up_Conv3D_1 = up_conv3D_block(filters_3D[5], filters_3D[4], (1, 2, 2))
        self.up_Conv3D_2 = up_conv3D_block(filters_3D[4] + filters_3D[4], filters_3D[3], (1, 2, 2))
        self.up_Conv3D_3 = up_conv3D_block(filters_3D[3] + filters_3D[3], filters_3D[2], (2, 2, 2))
        self.up_Conv3D_4 = up_conv3D_block(filters_3D[2] + filters_3D[2], filters_3D[1], (2, 2, 2))
        self.up_Conv3D_5 = up_conv3D_block(filters_3D[1] + filters_3D[1], filters_3D[0], (1, 2, 2))

        self.Conv3D_final = nn.Conv3d(filters_3D[0] + filters_3D[0], n_out, kernel_size=1, stride=1, padding=0)


    def forward(self, e_SA):
        # SA network's encoder
        e_SA_1 = self.Conv3D_1(e_SA)
        # print("E1:", e_SA_1.shape)

        e_SA = self.MaxPool3D_1(e_SA_1)
        # print("E2:", e_SA.shape)

        e_SA_2 = self.Conv3D_2(e_SA)
        # print("E3:", e_SA_2.shape)

        e_SA = self.MaxPool3D_2(e_SA_2)
        # print("E4:", e_SA.shape)

        e_SA_3 = self.Conv3D_3(e_SA)
        # print("E5:", e_SA_3.shape)

        e_SA = self.MaxPool3D_3(e_SA_3)
        # print("E6:", e_SA.shape)

        e_SA_4 = self.Conv3D_4(e_SA)
        # print("E7:", e_SA_4.shape)

        e_SA = self.MaxPool3D_4(e_SA_4)
        # print("E8:", e_SA.shape)

        e_SA_5 = self.Conv3D_5(e_SA)
        # print("E9:", e_SA_5.shape)

        e_SA = self.MaxPool3D_5(e_SA_5)
        # print("E10:", e_SA.shape)

        e_SA_6 = self.Conv3D_6(e_SA)
        # print("E11:", e_SA_6.shape)

        del (e_SA)

        # SA network's decoder
        d_SA = self.up_Conv3D_1(e_SA_6, e_SA_5.shape[2:])

        d_SA = torch.cat([e_SA_5, d_SA], dim=1)
        # print("D2:", d_SA.shape)

        d_SA = self.up_Conv3D_2(d_SA, e_SA_4.shape[2:])
        # print("D3:", d_SA.shape)
        
        d_SA = torch.cat([e_SA_4, d_SA], dim=1)
        # print("D4:", d_SA.shape)
        
        d_SA = self.up_Conv3D_3(d_SA, e_SA_3.shape[2:])
        # print("D5:", d_SA.shape)
        
        d_SA = torch.cat([e_SA_3, d_SA], dim=1)
        # print("D6:", d_SA.shape)
        
        d_SA = self.up_Conv3D_4(d_SA, e_SA_2.shape[2:])
        
        d_SA = torch.cat([e_SA_2, d_SA], dim=1)
        # print("D8:", d_SA.shape)
        
        d_SA = self.up_Conv3D_5(d_SA, e_SA_1.shape[2:])
        # print("D9:", d_SA.shape)
        
        d_SA = torch.cat([e_SA_1, d_SA], dim=1)
        # print("D10:", d_SA.shape)
        
        d_SA = self.Conv3D_final(d_SA)
        # print("D11:", d_SA.shape)

        del (e_SA_1, e_SA_2, e_SA_3, e_SA_4, e_SA_5)

        return d_SA