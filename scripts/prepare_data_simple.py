# Importing the necessary dependencies
import json
import os
import argparse
from os.path import abspath
import shutil
from glob import glob
import pandas as pd
from tqdm import tqdm
import nibabel as nib
import numpy as np
from nipype import Node, Workflow, Function
import nipype.interfaces.io as nio
from nipype.interfaces import fsl
from nipype.interfaces.fsl import BET, MeanImage, EpiReg, FLIRT, PrepareFieldmap, FUGUE, ConvertXFM
from nipype.interfaces.fsl.maths import ErodeImage
from nipype import SelectFiles
import argparse

def IntensityNormalization(in_file):
    """
    Intensity normalize the T1w input image
    """
    import nibabel as nib
    import numpy as np
    img = nib.load(in_file)
    data = img.get_fdata()
    image_histogram, bins = np.histogram(data[data != 0].flatten(), 256, density=True)
    cdf = image_histogram.cumsum()
    cdf = (256-1) * cdf / cdf[-1]
    image_equalized = np.interp(data.flatten(), bins[:-1], cdf).reshape(data.shape)
    out_path = f"{in_file}.norm.nii.gz"
    new_img = nib.Nifti1Image(image_equalized, img.affine)
    nib.save(new_img, out_path)
    return out_path

def main():
    parser = argparse.ArgumentParser(description="BOLD-T1w preprocess")
    parser.add_argument("--bold", required=True, help="4D BOLD time series")
    parser.add_argument("--t1w", required=True, help="T1w high-resolution structural image")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args()

    # Select nodes
    in_bold = Node(SelectFiles({"out_file": abspath(args.bold)}), name="in_functional_image")
    in_t1w  = Node(SelectFiles({"out_file": abspath(args.t1w)}),  name="in_anatomical_image")

    # Exports
    out_mc   = Node(nio.ExportFile(out_file=abspath(os.path.join(args.outdir, "bold_mc.nii.gz")), clobber=True), name='export_mc')
    out_t1w = Node(nio.ExportFile(out_file=abspath(os.path.join(args.outdir, "t1w_in_bold.nii.gz")), clobber=True), name='export_t1')

    # Define the preprocessing workflow
    wf = Workflow(name="simple_preproc")
    mc = Node(fsl.MCFLIRT(), name="motion_correct")
    mean_bold =  Node(fsl.maths.MeanImage(dimension='T'), name="mean_bold")
    anat_skull = Node(fsl.BET(frac=0.5, vertical_gradient=0.0), name="skullstrip_t1")
    epi2t1 = Node(fsl.epi.EpiReg(), name="epi2t1")
    inv_epi2t1 = Node(fsl.utils.ConvertXFM(invert_xfm=True), name="inv_epi2t1")
    anat_trans = Node(fsl.FLIRT(apply_xfm=True), name="anat_trans")
    intensity_norm_anat = Node(Function(function=IntensityNormalization, input_names=["in_file"], output_names=["out_file"]), name="intensity_norm_anat")
    
    wf.connect([
        # BOLD -> MCFLIRT -> Export MC
        (in_bold, mc, [("out_file", "in_file")]),
        (mc, out_mc, [("out_file", "in_file")]),

        # MCFLIRT -> MeanImage
        (mc, mean_bold, [("out_file", "in_file")]),

        # T1w -> BET
        (in_t1w, anat_skull, [("out_file", "in_file")]),

        # Mean BOLD -> epi2t1 moving, skull -> epi2t1 reference
        # (epi to struct)
        (mean_bold, epi2t1, [("out_file", "epi")]),
        (anat_skull, epi2t1, [("out_file", "t1_brain")]),
        (in_t1w, epi2t1, [("out_file", "t1_head")]),

        # epi2t1 -> invert
        # (struct to epi)
        (epi2t1, inv_epi2t1, [("epi2str_mat", "in_file")]),

        # Skull -> intensity normalize
        (anat_skull, intensity_norm_anat, [("out_file", "in_file")]),

        # normalized T1 + inverted matrix + mean BOLD reference → FLIRT apply_xfm → Export T1 in BOLD
        (intensity_norm_anat, anat_trans, [("out_file", "in_file")]),
        (inv_epi2t1, anat_trans, [("out_file", "in_matrix_file")]),
        (mean_bold, anat_trans, [("out_file", "reference")]),
        (anat_trans, out_t1w, [("out_file", "in_file")]),
    ])

    # Run it
    wf.run()

if __name__ == "__main__":
    main()