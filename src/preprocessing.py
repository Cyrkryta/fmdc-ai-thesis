# src/preprocessing.py
import os
import tempfile
from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces import fsl
from nipype.interfaces.fsl import MCFLIRT, MeanImage, BET, EpiReg, FLIRT
from nipype.interfaces.fsl.utils import ConvertXFM
from nipype.interfaces.utility import Function
from nipype.interfaces.io import ExportFile
import shutil

def IntensityNormalization(in_file):
    """Intensity‐normalize a skull‐stripped T1w image."""
    import nibabel as nib, numpy as np
    img  = nib.load(in_file)
    data = img.get_fdata()
    hist, bins = np.histogram(data[data!=0].flatten(), 256, density=True)
    cdf = hist.cumsum(); cdf = (256-1)*cdf/cdf[-1]
    eq  = np.interp(data.flatten(), bins[:-1], cdf).reshape(data.shape)
    out = f"{in_file}.norm.nii.gz"
    nib.save(nib.Nifti1Image(eq, img.affine), out)
    return out

def preprocess_input_data(bold_fn: str, t1w_fn: str, outdir: str):
    """
    Runs your original 1→7 Nipype steps and writes two files into outdir:
      - bold_mc.nii.gz
      - t1w_in_bold.nii.gz
    Returns a dict with those paths.
    """
    # Create the workflow
    os.makedirs(outdir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="ni1_")
    wf  = Workflow(name="simple_preproc", base_dir=tmp)

    # Create the nodes
    inp = Node(IdentityInterface(fields=["bold", "t1w", "outdir"]), name="inputnode")   # Create the nodes
    inp.inputs.bold = bold_fn
    inp.inputs.t1w = t1w_fn
    inp.inputs.outdir = outdir
    mc = Node(MCFLIRT(), name="motion_correct")                                         # Motion correction and export
    exp_mc = Node(ExportFile(clobber=True), name="export_mc")
    exp_mc.inputs.out_file = os.path.join(outdir, "bold_mc.nii.gz")
    mean_bold = Node(MeanImage(dimension="T"), name="mean_bold")                             # Mean image of BOLD
    anat_skull = Node(fsl.BET(frac=0.5, vertical_gradient=0.0), name="skullstrip_t1")   # Anatomical skull-stripping
    epi2t1 = Node(EpiReg(), name="epi2t1")                                              # EPI registration
    inv_epi2t1 = Node(fsl.utils.ConvertXFM(invert_xfm=True), name="inv_epi2t1")         # Invert the EPI to T1w transform
    anat_trans = Node(fsl.FLIRT(apply_xfm=True), name="anat_trans")
    intensity_norm_anat = Node(Function(function=IntensityNormalization, input_names=["in_file"], output_names=["out_file"]), name="intensity_norm_anat")
    exp_t1 = Node(ExportFile(clobber=True), name="export_t1")
    exp_t1.inputs.out_file = os.path.join(outdir, "t1w_in_bold.nii.gz")

    # Connect the nodes
    wf.connect([
        (inp, mc, [("bold", "in_file")]),                               # BOLD -> MCFLIRT 
        (mc, exp_mc, [("out_file", "in_file")]),
        (mc, mean_bold, [("out_file", "in_file")]),                     # MCFLIRT -> MeanImage
        (inp, anat_skull, [("t1w", "in_file")]),                        # T1w -> BET
        (mean_bold, epi2t1, [("out_file", "epi")]),                     # Mean BOLD -> epi2t1 moving, skull -> epi2t1 reference
        (anat_skull, epi2t1, [("out_file", "t1_brain")]),
        (inp, epi2t1, [("t1w", "t1_head")]),
        (epi2t1, inv_epi2t1, [("epi2str_mat", "in_file")]),             # epi2t1 -> invert (struct to epi)
        (anat_skull, intensity_norm_anat, [("out_file", "in_file")]),   # Skull -> intensity normalize
        (intensity_norm_anat, anat_trans, [("out_file", "in_file")]),   # Transform T1w image
        (inv_epi2t1, anat_trans, [("out_file", "in_matrix_file")]),
        (mean_bold, anat_trans, [("out_file", "reference")]),
        (anat_trans, exp_t1, [("out_file", "in_file")])                 # Export the resampled T1
    ])

    # Run the workflow
    wf.run()

    # Clean up the temporary directory after workflow execution
    shutil.rmtree(tmp, ignore_errors=True)

    # Return the output file paths
    return {
        "bold_mc": os.path.join(outdir, "bold_mc.nii.gz"),
        "t1w_in_bold": os.path.join(outdir, "t1w_in_bold.nii.gz")
    }