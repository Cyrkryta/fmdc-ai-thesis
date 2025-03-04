import json
import os
from os.path import abspath
import shutil
from glob import glob

import pandas as pd
from tqdm import tqdm
from nipype import Node, Workflow, Function
import nipype.interfaces.io as nio
from nipype.interfaces import fsl
from nipype import SelectFiles


def GetMedianTF(in_file):
    import nibabel as nib
    return int(nib.load(in_file).header['dim'][4] / 2)


def SubtractFive(in_value):
    return in_value - 5


def IntensityNormalization(in_file):
    import nibabel as nib
    import numpy as np

    img = nib.load(in_file)
    data = img.get_fdata()

    image_histogram, bins = np.histogram(data[data != 0].flatten(), 256, density=True)
    cdf = image_histogram.cumsum()
    cdf = (256-1) * cdf / cdf[-1]

    image_equalized = np.interp(data.flatten(), bins[:-1], cdf).reshape(data.shape)

    out_path = f'{in_file}.norm.nii.gz'
    new_img = nib.Nifti1Image(image_equalized, img.affine)
    nib.save(new_img, out_path)

    return out_path


# Method for converting the subject
def _convert_subject(subject_path, dataset, fsl_dir):
    # Retrieve the full normalized path, retrieve the subject
    subject_path = os.path.normpath(subject_path)
    subject = os.path.basename(subject_path)

    # Retrieve the different input paths
    # Convert to Node to be able to connect to other in the preprocessing pipeline
    in_anatomical_image = Node(SelectFiles({"out_file": abspath(dataset["paths"]["anatomical"].format(subject=subject, subject_path=subject_path))}), name="in_anatomical_image")
    in_magnitude_image = Node(SelectFiles({"out_file": abspath(dataset["paths"]["fmap"]["magnitude"].format(subject=subject, subject_path=subject_path))}), name="in_magnitude_image")
    in_phase_image = Node(SelectFiles({"out_file": abspath(dataset["paths"]["fmap"]["phasediff"].format(subject=subject, subject_path=subject_path))}), name="in_phase_image")
    in_functional_image = Node(SelectFiles({"out_file": abspath(dataset["paths"]["functional"].format(subject=subject, subject_path=subject_path))}), name="in_functional_image")

    # Creating output directory for the designated subject
    # (is this even used though)
    output_dir = os.path.join(subject_path, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # FILE NODE PREPARATION
    # Objects for writing node to files
    # Written to the results output dir
    out_field_map = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "field_map.nii.gz")), clobber=True), name="out_field_map")
    out_b0_d = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "b0_d.nii.gz")), clobber=True), name="out_b0_d")
    out_b0_mask = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "b0_mask.nii.gz")), clobber=True), name="out_b0_mask")
    out_b0_u = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "b0_u.nii.gz")), clobber=True), name="out_b0_u")
    out_t1w = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "T1w.nii.gz")), clobber=True), name="out_t1w")
    out_mni_warp = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "T1w_to_MNI_warp.nii.gz")), clobber=True), name="out_mni_warp")
    out_func_to_struct = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "func2struct.mat")), clobber=True), name="out_func_to_struct")

    # WORKFLOW NODES PREPARATION
    # T1w image
    # Skullstrip of the anatomical images
    skullstrip_t1w = Node(fsl.BET(frac=0.5, vertical_gradient=0.0, mask=True), name="skullstrip_t1w")

    # Magnitude image
    # Skullstrip the magnitude image
    skullstrip_magnitude = Node(fsl.BET(), name="skullstrip_magnitude")
    # Removes edge artifacts from the model
    erode_magnitude = Node(fsl.maths.ErodeImage(), name="erode_magnitude")

    # Fieldmap Image
    # Node for running prepare fieldmap
    prepare_fieldmap = Node(fsl.PrepareFieldmap(delta_TE=2.46), name="prepare_fieldmap")

    # BOLD Image
    # Find the median time index (e.g. series of 818 points, median 409)
    median_tf = Node(Function(function=GetMedianTF, input_names=["in_file"], output_names=["out_value"]), name="median_tf")
    # Node for doing a single extraction (going to be at the median tf)
    extract_roi_functional = Node(fsl.ExtractROI(t_size=1), name="extract_roi_functional")
    # Median minus five for create a starting point to extract time-series
    median_tf_minus_five = Node(Function(function=SubtractFive, input_names=["in_value"], output_names=["out_value"]), name="median_tf_minus_five")
    # Extracting a time-series of 10 steps (probably starting at median-5)
    extract_roi_functional_10 = Node(fsl.ExtractROI(t_size=10), name="extract_roi_functional_10")

    # Correction
    # Node for performing the correction
    fugue_correction = Node(fsl.FUGUE(dwell_time=dataset["echoSpacing"], smooth3d=3, unwarp_direction="y-"), name="fugue_correction")

    # Registration
    # Register functional image to structural image
    register_functional_to_t1 = Node(fsl.epi.EpiReg(), name="register_functional_to_t1")
    # Node for computing the inverse mapping
    invert_registration_matrix = Node(fsl.utils.ConvertXFM(invert_xfm=True), name="invert_registration_matrix")

    # Additional processing nodes
    # Skullstrip to create a brain extracted version of the short time-series
    skullstrip_functional_10 = Node(fsl.BET(functional=True, mask=True), name="skullstrip_functional_10")
    # Node for skullstripping the unwarped functional image
    skullstrip_unwarped_functional = Node(fsl.BET(functional=True), name="skullstrip_unwarped_functional")
    # Node for using FLIRT to register one image on to
    transform_t1 = Node(fsl.FLIRT(apply_xfm=True), name="transform_t1")
    # Normalizing T1 intensities
    intensity_normalize_t1 = Node(Function(function=IntensityNormalization, input_names=["in_file"], output_names=["out_file"]), name="intensity_normalize_t1")
    # Registration of the T1 image to a standard template
    # Affine registration
    transform_t1_to_mni = Node(fsl.FLIRT(dof=12, reference=f"{fsl_dir}/data/standard/MNI152_T1_2mm_brain.nii.gz"), name="transform_t1_to_mni")
    # Non-linear registration / transformation
    register_t1_to_mni = Node(fsl.FNIRT(config_file="T1_2_MNI152_2mm", ref_file=f"{fsl_dir}/data/standard/MNI152_T1_2mm_brain.nii.gz", field_file=True), name="register_t1_to_mni")

    # WORKFLOW
    # Defining the workflow
    workflow = Workflow(name="prepare_subject")

    # (64, 64, 36): The raw magnitude image is provided to skull stripping.
    workflow.connect(in_magnitude_image, "out_file", skullstrip_magnitude, "in_file")
    # (64, 64, 36): The skull-stripped magnitude image is then eroded to refine the brain region..
    workflow.connect(skullstrip_magnitude, "out_file", erode_magnitude, "in_file")
    # (64, 64, 36): The eroded magnitude image is passed as the "in_magnitude" input for field map preparation.
    workflow.connect(erode_magnitude, "out_file", prepare_fieldmap, "in_magnitude")
    # (64, 64, 36): The phase image (raw input) is provided as the "in_phase" input for field map preparation.
    workflow.connect(in_phase_image, "out_file", prepare_fieldmap, "in_phase")
    # (64, 64, 36): The computed field map is exported to a file.
    workflow.connect(prepare_fieldmap, "out_fieldmap", out_field_map, "in_file")

    # (64, 64, 36, 818): The full 4D BOLD image is sent to compute the median time frame.
    workflow.connect(in_functional_image, "out_file", median_tf, "in_file")
    # (64, 64, 36, 818): The same full BOLD image is sent to extract a single 3D volume (median frame).
    workflow.connect(in_functional_image, "out_file", extract_roi_functional, "in_file")
    # (64, 64, 36, 818): Also, the full BOLD image is used to extract a short 4D segment (e.g., 10 volumes).
    workflow.connect(in_functional_image, "out_file", extract_roi_functional_10, "in_file")
    # The computed median time point (e.g., 409) is provided to extract the single 3D volume from the BOLD image
    workflow.connect(median_tf, "out_value", extract_roi_functional, "t_min")
    # The same median value is sent to a node that subtracts 5 (e.g., 409 → 404) for the multi-volume extraction.
    workflow.connect(median_tf, "out_value", median_tf_minus_five, "in_value")
    # The adjusted time point (e.g., 404) is provided to extract a 10-volume ROI.
    workflow.connect(median_tf_minus_five, "out_value", extract_roi_functional_10, "t_min")

    # (64, 64, 36, 10): The 10-volume ROI is provided to the distortion correction node (FUGUE).
    workflow.connect(extract_roi_functional_10, "roi_file", fugue_correction, "in_file")
    # (64, 64, 36): The computed field map is also provided to FUGUE for distortion correction.
    workflow.connect(prepare_fieldmap, "out_fieldmap", fugue_correction, "fmap_in_file")

    # (64, 64, 36): The single extracted 3D functional volume is sent to register with the anatomical T1.
    workflow.connect(extract_roi_functional, "roi_file", register_functional_to_t1, "epi")

    # (224, 256, 256): The anatomical T1 image is provided to skull stripping.
    workflow.connect(in_anatomical_image, "out_file", skullstrip_t1w, "in_file")
    # (224, 256, 256): The skull-stripped T1 (brain only) is given as "t1_brain" for registration.
    workflow.connect(skullstrip_t1w, "out_file", register_functional_to_t1, "t1_brain")
    # (224, 256, 256): The original T1 (with head) is provided as "t1_head" for registration.
    workflow.connect(in_anatomical_image, "out_file", register_functional_to_t1, "t1_head")

    # (Matrix): The functional-to-anatomical transformation matrix (e.g., 4×4 affine) is exported to a file.
    workflow.connect(register_functional_to_t1, "epi2str_mat", out_func_to_struct, "in_file")
    # (Matrix): The same transformation matrix is sent to a node that computes its inverse.
    workflow.connect(register_functional_to_t1, "epi2str_mat", invert_registration_matrix, "in_file")

    # (64, 64, 36, 10): The 10-volume ROI is skull-stripped to remove non-brain tissue.
    workflow.connect(extract_roi_functional_10, "roi_file", skullstrip_functional_10, "in_file")
    # (64, 64, 36, 10): The resulting skull-stripped functional data (distorted) is exported.
    workflow.connect(skullstrip_functional_10, "out_file", out_b0_d, "in_file")
    # (64, 64, 36, 10): The brain mask from the skull-stripping is also exported.
    workflow.connect(skullstrip_functional_10, "mask_file", out_b0_mask, "in_file")

    # (64, 64, 36, 10): The unwarped (distortion-corrected) functional data from FUGUE is skull-stripped.
    workflow.connect(fugue_correction, "unwarped_file", skullstrip_unwarped_functional, "in_file")
    # (64, 64, 36, 10): The resulting skull-stripped, unwarped functional data is exported.
    workflow.connect(skullstrip_unwarped_functional, "out_file", out_b0_u, "in_file")

    # (224, 256, 256): The skull-stripped T1 image is intensity-normalized.
    workflow.connect(skullstrip_t1w, "out_file", intensity_normalize_t1, "in_file")
    # (224, 256, 256): The normalized T1 is then transformed into the space of the functional image.
    workflow.connect(intensity_normalize_t1, "out_file", transform_t1, "in_file")
    # (64, 64, 36, 10): The skull-stripped, unwarped functional image is used as the reference for the T1 transformation.
    workflow.connect(skullstrip_unwarped_functional, "out_file", transform_t1, "reference")
    # (Matrix): The inverted registration matrix is provided to correctly map the T1 into functional space.
    workflow.connect(invert_registration_matrix, "out_file", transform_t1, "in_matrix_file")
    # (Transformed T1): The T1 image, now resampled to functional space (e.g., ~64, 64, 36), is exported
    workflow.connect(transform_t1, "out_file", out_t1w, "in_file")

    # (224, 256, 256): The normalized T1 image is used for the initial affine transformation to MNI space.
    workflow.connect(intensity_normalize_t1, "out_file", transform_t1_to_mni, "in_file")
    #  (224, 256, 256): The normalized T1 image is also provided for nonlinear registration to MNI space
    workflow.connect(intensity_normalize_t1, "out_file", register_t1_to_mni, "in_file")
    # (Matrix): The affine transformation matrix from the FLIRT step is passed to the FNIRT node.
    workflow.connect(transform_t1_to_mni, "out_matrix_file", register_t1_to_mni, "affine_file")
    # (Warp Field): The final nonlinear warp field (mapping T1 to MNI space, e.g., dimensions might be ~91, 109, 91) is exported.
    workflow.connect(register_t1_to_mni, "field_file", out_mni_warp, "in_file")

    # Run the entire workflow
    workflow.run()


def _copy_output_to_new_dir(subject_path, subject_output):
    if not os.path.exists(subject_output):
        os.makedirs(subject_output)

    shutil.copytree(os.path.join(subject_path, "results"), subject_output, dirs_exist_ok=True)


def _convert_all_datasets(dataset_root, output_root, fsl_dir):
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    with open(f'{dataset_root}/training_datasets.json', 'r') as ds_file:
        datasets = json.load(ds_file)

    for dataset_name, dataset in datasets.items():
        print(f'Converting dataset "{dataset_name}"...')

        dataset_output = os.path.join(output_root, dataset_name)
        if not os.path.exists(dataset_output):
            os.makedirs(dataset_output)

        for subject_path in tqdm(glob(f'{dataset_root}/{dataset_name}/sub-*/')):
            subject = os.path.basename(os.path.normpath(subject_path))
            subject_output = os.path.join(dataset_output, subject)

            all_files_exist = os.path.isfile(os.path.join(subject_output, 'T1w.nii.gz')) and \
                os.path.isfile(os.path.join(subject_output, 'b0_d.nii.gz')) and \
                os.path.isfile(os.path.join(subject_output, 'b0_u.nii.gz')) and \
                os.path.isfile(os.path.join(subject_output, 'b0_mask.nii.gz')) and \
                os.path.isfile(os.path.join(subject_output, 'T1w_to_MNI_warp.nii.gz')) and \
                os.path.isfile(os.path.join(subject_output, 'field_map.nii.gz')) and \
                os.path.isfile(os.path.join(subject_output, 'func2struct.mat'))
            subject_excluded = subject in dataset['excludedSubjects']

            if all_files_exist or subject_excluded:
                continue
            else:
                print(f'Processing subject {subject}...')
                _convert_subject(subject_path, dataset, fsl_dir)
                _copy_output_to_new_dir(subject_path, subject_output)

        participants_df = pd.read_csv(os.path.join(f'{dataset_root}/{dataset_name}', 'participants.tsv'), sep='\t')
        for dropped_subject in dataset['excludedSubjects']:
            participants_df = participants_df.drop(participants_df[participants_df.participant_id == dropped_subject].index)
        participants_df.to_csv(os.path.join(dataset_output, 'participants.tsv'), sep='\t')

        dataset_meta = {
            'echoSpacing': dataset['echoSpacing']
        }
        with open(os.path.join(dataset_output, 'dataset_meta.json'), "w") as meta_file:
            meta_file.write(json.dumps(dataset_meta, indent=4))


if __name__ == '__main__':
    """
    Given a set of raw OpenNeuro datasets along with JSON metadata, transform them into a representation that can be
    directly used for training and inferring distortion correction models.
    """

    fsl_dir = '/home/mlc/fsl'
    dataset_root = '/home/mlc/dev/fmdc/downloads/openneuro-datasets'
    output_root = '/home/mlc/dev/fmdc/downloads/openneuro-datasets/preprocessed'
    _convert_all_datasets(dataset_root=dataset_root, output_root=output_root, fsl_dir=fsl_dir)
