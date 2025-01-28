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


def GetMedianTF(in_file):
    import nibabel as nib
    return int(nib.load(in_file).header['dim'][4] / 2)


def SubtractFive(in_value):
    return in_value - 5


class UNet3DFieldmap(pl.LightningModule):
    """
    A three-dimensional U-Net that is trained to predict the fieldmap of an fMRI scan rather than the undistorted scan
    directly.
    """

    def __init__(self):
        super().__init__()
        self.model = UNet3D_2Module(2, 1)

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        img, _, _, fieldmap, _, _ = batch
        out = self(img)
        loss = self.compute_loss(out, fieldmap)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, b0_u, mask, fieldmap, affine, echo_spacing = batch
        out = self(img)
        loss = self.compute_loss(out, fieldmap)
        if batch_idx == 0:
            self._log_images(out, img, b0_u, mask, fieldmap, affine, echo_spacing)
        self.log('val_loss', loss)
        return loss

    def _undistort_b0(self, b0_d, fieldmap, affine_b0d, affine_fieldmap, echo_spacing):
        with tempfile.TemporaryDirectory() as directory:
            b0_d = np.transpose(b0_d.cpu().detach().numpy(), axes=(1, 2, 0))
            b0_d = np.repeat(b0_d[:, :, :, None], 10, axis=3)
            b0_d_image = nib.Nifti1Image(b0_d, affine_b0d)
            nib.save(b0_d_image, os.path.join(directory, 'b0_d.nii.gz'))
            fieldmap = fieldmap.cpu().detach().numpy()[0]
            fieldmap = np.transpose(fieldmap, axes=(1, 2, 0))
            fieldmap_image = nib.Nifti1Image(fieldmap, affine_fieldmap)
            nib.save(fieldmap_image, os.path.join(directory, 'field_map.nii.gz'))

            in_b0d = Node(SelectFiles({"out_file": abspath(os.path.join(directory, 'b0_d.nii.gz'))}), name="in_b0d")
            in_fieldmap = Node(SelectFiles({"out_file": abspath(os.path.join(directory, 'field_map.nii.gz'))}), name="in_fieldmap")
            out_b0_u = Node(nio.ExportFile(out_file=abspath(os.path.join(directory, "b0_u.nii.gz")), clobber=True), name="out_b0_u")

            fugue_correction = Node(fsl.FUGUE(dwell_time=echo_spacing, smooth3d=3, unwarp_direction="y-"), name="fugue_correction")

            workflow = Workflow(name="undistort_subject")

            workflow.connect(in_b0d, "out_file", fugue_correction, "in_file")
            workflow.connect(in_fieldmap, "out_file", fugue_correction, "fmap_in_file")
            workflow.connect(fugue_correction, "unwarped_file", out_b0_u, "in_file")

            workflow.run()

            out = nib.load(os.path.join(directory, 'b0_u.nii.gz')).get_fdata()

        return out

    def _log_images(self, out, img, b0_u, mask, fieldmap, affine, echo_spacing):
        sample_idx = 0
        slice_idx = 18

        t1 = img[sample_idx][1][slice_idx]
        b0_d = img[sample_idx][0][slice_idx]
        b0_u = b0_u[sample_idx][0][slice_idx]
        fieldmap = fieldmap[sample_idx][0][slice_idx]
        affine = affine[sample_idx]
        echo_spacing = echo_spacing[sample_idx]

        wandb.log({
            'epoch': self.current_epoch,
            't1': wandb.Image(t1, caption="T1w"),
            'b0_d': wandb.Image(b0_d, caption="B0 Distorted"),
            'b0_u': wandb.Image(b0_u, caption="Ground Truth (B0 Undistorted)"),
            'fieldmap': wandb.Image(fieldmap, caption="Ground Truth Fieldmap"),
            'out': wandb.Image(out[sample_idx][0][slice_idx], caption="Model Output Fieldmap")
        })

    def compute_loss(self, out, fieldmap):
        return F.mse_loss(out, fieldmap)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5)
        return optimizer


class conv3D_block(nn.Module):

    def __init__(self, in_ch, out_ch):

        super(conv3D_block, self).__init__()

        self.conv3D = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv3D(x)
        return x

class up_conv3D_block(nn.Module):

    def __init__(self, in_ch, out_ch, scale_tuple):

        super(up_conv3D_block, self).__init__()

        self.up_conv3D = nn.Sequential(
            nn.Upsample(scale_factor=scale_tuple, mode='trilinear'),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True), # increasing the depth by adding one below
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
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
        d_SA = self.up_Conv3D_1(e_SA_6)
        # print("D1:", d_SA.shape)
        d_SA = torch.cat([e_SA_5, d_SA], dim=1)
        # print("D2:", d_SA.shape)
        d_SA = self.up_Conv3D_2(d_SA)
        # print("D3:", d_SA.shape)
        d_SA = torch.cat([e_SA_4, d_SA], dim=1)
        # print("D4:", d_SA.shape)
        d_SA = self.up_Conv3D_3(d_SA)
        # print("D5:", d_SA.shape)
        d_SA = torch.cat([e_SA_3, d_SA], dim=1)
        # print("D6:", d_SA.shape)
        d_SA = self.up_Conv3D_4(d_SA)
        # print("D7:", d_SA.shape)
        d_SA = torch.cat([e_SA_2, d_SA], dim=1)
        # print("D8:", d_SA.shape)
        d_SA = self.up_Conv3D_5(d_SA)
        # print("D9:", d_SA.shape)
        d_SA = torch.cat([e_SA_1, d_SA], dim=1)
        # print("D10:", d_SA.shape)
        d_SA = self.Conv3D_final(d_SA)
        # print("D11:", d_SA.shape)

        del (e_SA_1, e_SA_2, e_SA_3, e_SA_4, e_SA_5)

        return d_SA
