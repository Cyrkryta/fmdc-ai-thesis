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

    # Forward layer of the mode
    def forward(self, img):
        # print("\nForward\n")
        return self.model(img)

    # Training step
    def training_step(self, batch, batch_idx):
        # print("\nTraining step\n")
        # Retrieve the img and batch
        img_data = batch["img_data"]
        fieldmap = batch["fieldmap"]

        # Compute fieldmap
        out = self(img_data)

        # Compute the training loss
        train_loss = self.compute_loss(out, fieldmap)

        # Log the loss
        self.log("train_loss", train_loss, on_step=False, on_epoch=True)

        # Return the loss
        return train_loss


    # Validation step
    def validation_step(self, batch, batch_idx):
        # print("\nValidation step")
        # Retrieve elemens from the batch
        img_data = batch["img_data"]
        fieldmap = batch["fieldmap"]
        # affine = batch["fieldmap_affine"]
        # echo_spacing = batch["echo_spacing"]
        # unwarp_direction = batch["unwarp_direction"]
        # mask = batch["mask"]
        # b0_u = batch["b0_u"]

        # Compute the fieldmap estimate
        out = self(img_data)

        # Compute the loss
        val_loss = self.compute_loss(out, fieldmap)

        # Log the validation loss
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)

        # Log first sample images in each batch to W&B
        if batch_idx == 0:
            self._log_images(out, img_data, fieldmap)
            # self._log_images(out, img_data, fieldmap, affine, echo_spacing)

        # Return the loss
        return val_loss


    # Undistort b0
    def _undistort_b0(self, b0_d, fieldmap, affine_b0d, affine_fieldmap, echo_spacing, unwarp_direction):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as directory:
            # Move distorted volume to CPU, detach, and convert to a numpy array (repeated 10 times?)
            b0_d = np.transpose(b0_d.cpu().detach().numpy(), axes=(1, 2, 0))
            b0_d = np.repeat(b0_d[:, :, :, None], 10, axis=3)
            # Converting numby array to an nifti image
            b0_d_image = nib.Nifti1Image(b0_d, affine_b0d)
            # Save the distorted file
            nib.save(b0_d_image, os.path.join(directory, 'b0_d.nii.gz'))
            # Detach the fieldmap, convert to numpy (only the first)
            fieldmap = fieldmap.cpu().detach().numpy()[0]
            # Transpose and save the fieldmap image
            fieldmap = np.transpose(fieldmap, axes=(1, 2, 0))
            fieldmap_image = nib.Nifti1Image(fieldmap, affine_fieldmap)
            nib.save(fieldmap_image, os.path.join(directory, 'field_map.nii.gz'))

            # Create necessary nodes for undistortion.
            in_b0d = Node(SelectFiles({"out_file": abspath(os.path.join(directory, 'b0_d.nii.gz'))}), name="in_b0d")
            in_fieldmap = Node(SelectFiles({"out_file": abspath(os.path.join(directory, 'field_map.nii.gz'))}), name="in_fieldmap")
            out_b0_u = Node(nio.ExportFile(out_file=abspath(os.path.join(directory, "b0_u.nii.gz")), clobber=True), name="out_b0_u")
            fugue_correction = Node(fsl.FUGUE(dwell_time=echo_spacing, smooth3d=3, unwarp_direction=unwarp_direction), name="fugue_correction")


            # Perform the fugue correction
            workflow = Workflow(name="undistort_subject")
            workflow.connect(in_b0d, "out_file", fugue_correction, "in_file")
            workflow.connect(in_fieldmap, "out_file", fugue_correction, "fmap_in_file")
            workflow.connect(fugue_correction, "unwarped_file", out_b0_u, "in_file")
            workflow.run()

            # Load the undistorted file from the temporary directory
            out = nib.load(os.path.join(directory, 'b0_u.nii.gz')).get_fdata

        # Return the undistorted file
        return out

    # Logging images
    # def _log_images(self, out, img, b0_u, mask, fieldmap, affine, echo_spacing):
    # def _log_images(self, out, img, fieldmap, affine, echo_spacing):
    def _log_images(self, out, img, fieldmap):
        # Pick a smple and slice
        sample_idx = 0
        slice_idx = img[sample_idx].shape[1] // 2

        # Retriieve the sampled images
        t1 = img[sample_idx][1][slice_idx]
        b0_d = img[sample_idx][0][slice_idx]
        # b0_u = b0_u[sample_idx][0][slice_idx]
        fieldmap = fieldmap[sample_idx][0][slice_idx]
        # affine = affine[sample_idx]
        # echo_spacing = echo_spacing[sample_idx]

        # Log the sample images to wandb
        wandb.log({
            'epoch': self.current_epoch,
            't1': wandb.Image(t1, caption="T1w"),
            'b0_d': wandb.Image(b0_d, caption="B0 Distorted"),
            # 'b0_u': wandb.Image(b0_u, caption="Ground Truth (B0 Undistorted)"),
            'fieldmap': wandb.Image(fieldmap, caption="Ground Truth Fieldmap"),
            'out': wandb.Image(out[sample_idx][0][slice_idx], caption="Model Output Fieldmap")
        })

    # Function for defining and computing the loss function
    def compute_loss(self, out, fieldmap):
        # print("\nCompute loss\n")
        computed_loss = F.mse_loss(out, fieldmap)
        return computed_loss

    # Configuration of the optimizer
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

# """
# Class:
# Definition of 3D up-convolutional block
# """
# class conv3D_block(nn.Module):

#     def __init__(self, in_ch, out_ch):

#         super(conv3D_block, self).__init__()

#         self.conv3D = nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
#             nn.InstanceNorm3d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
#             nn.InstanceNorm3d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.conv3D(x)
#         return x

# class up_conv3D_block(nn.Module):

#     def __init__(self, in_ch, out_ch, scale_tuple):

#         super(up_conv3D_block, self).__init__()

#         self.up_conv3D = nn.Sequential(
#             nn.Upsample(scale_factor=scale_tuple, mode='trilinear'),
#             nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
#             nn.InstanceNorm3d(out_ch),
#             nn.ReLU(inplace=True), # increasing the depth by adding one below
#             nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
#             nn.InstanceNorm3d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.up_conv3D(x)
#         return x

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
        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
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
        # d_SA = self.up_Conv3D_1(e_SA_6)
        d_SA = self.up_Conv3D_1(e_SA_6, e_SA_5.shape[2:])
        # print("D1:", d_SA.shape)

        d_SA = torch.cat([e_SA_5, d_SA], dim=1)
        # print("D2:", d_SA.shape)

        # d_SA = self.up_Conv3D_2(d_SA)
        d_SA = self.up_Conv3D_2(d_SA, e_SA_4.shape[2:])
        # print("D3:", d_SA.shape)
        
        d_SA = torch.cat([e_SA_4, d_SA], dim=1)
        # print("D4:", d_SA.shape)
        
        # d_SA = self.up_Conv3D_3(d_SA)
        d_SA = self.up_Conv3D_3(d_SA, e_SA_3.shape[2:])
        # print("D5:", d_SA.shape)
        
        d_SA = torch.cat([e_SA_3, d_SA], dim=1)
        # print("D6:", d_SA.shape)
        
        # d_SA = self.up_Conv3D_4(d_SA)
        d_SA = self.up_Conv3D_4(d_SA, e_SA_2.shape[2:])
        # print("D7:", d_SA.shape)
        
        d_SA = torch.cat([e_SA_2, d_SA], dim=1)
        # print("D8:", d_SA.shape)
        
        # d_SA = self.up_Conv3D_5(d_SA)
        d_SA = self.up_Conv3D_5(d_SA, e_SA_1.shape[2:])
        # print("D9:", d_SA.shape)
        
        d_SA = torch.cat([e_SA_1, d_SA], dim=1)
        # print("D10:", d_SA.shape)
        
        d_SA = self.Conv3D_final(d_SA)
        # print("D11:", d_SA.shape)

        del (e_SA_1, e_SA_2, e_SA_3, e_SA_4, e_SA_5)
    
        # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

        return d_SA