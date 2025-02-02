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


def _convert_subject(subject_path, dataset, fsl_dir):
    subject_path = os.path.normpath(subject_path)
    subject = os.path.basename(subject_path)

    in_anatomical_image = Node(SelectFiles({"out_file": abspath(dataset["paths"]["anatomical"].format(subject=subject, subject_path=subject_path))}), name="in_anatomical_image")
    in_magnitude_image = Node(SelectFiles({"out_file": abspath(dataset["paths"]["fmap"]["magnitude"].format(subject=subject, subject_path=subject_path))}), name="in_magnitude_image")
    in_phase_image = Node(SelectFiles({"out_file": abspath(dataset["paths"]["fmap"]["phasediff"].format(subject=subject, subject_path=subject_path))}), name="in_phase_image")
    in_functional_image = Node(SelectFiles({"out_file": abspath(dataset["paths"]["functional"].format(subject=subject, subject_path=subject_path))}), name="in_functional_image")

    output_dir = os.path.join(subject_path, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    out_field_map = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "field_map.nii.gz")), clobber=True), name="out_field_map")
    out_b0_d = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "b0_d.nii.gz")), clobber=True), name="out_b0_d")
    out_b0_mask = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "b0_mask.nii.gz")), clobber=True), name="out_b0_mask")
    out_b0_u = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "b0_u.nii.gz")), clobber=True), name="out_b0_u")
    out_t1w = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "T1w.nii.gz")), clobber=True), name="out_t1w")
    out_mni_warp = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "T1w_to_MNI_warp.nii.gz")), clobber=True), name="out_mni_warp")
    out_func_to_struct = Node(nio.ExportFile(out_file=abspath(os.path.join(output_dir, "func2struct.mat")), clobber=True), name="out_func_to_struct")

    skullstrip_t1w = Node(fsl.BET(frac=0.5, vertical_gradient=0.0, mask=True), name="skullstrip_t1w")
    skullstrip_magnitude = Node(fsl.BET(), name="skullstrip_magnitude")
    erode_magnitude = Node(fsl.maths.ErodeImage(), name="erode_magnitude")
    prepare_fieldmap = Node(fsl.PrepareFieldmap(delta_TE=2.46), name="prepare_fieldmap")
    median_tf = Node(Function(function=GetMedianTF, input_names=["in_file"], output_names=["out_value"]), name="median_tf")
    extract_roi_functional = Node(fsl.ExtractROI(t_size=1), name="extract_roi_functional")
    median_tf_minus_five = Node(Function(function=SubtractFive, input_names=["in_value"], output_names=["out_value"]), name="median_tf_minus_five")
    extract_roi_functional_10 = Node(fsl.ExtractROI(t_size=10), name="extract_roi_functional_10")
    fugue_correction = Node(fsl.FUGUE(dwell_time=dataset["echoSpacing"], smooth3d=3, unwarp_direction="y-"), name="fugue_correction")
    register_functional_to_t1 = Node(fsl.epi.EpiReg(), name="register_functional_to_t1")
    invert_registration_matrix = Node(fsl.utils.ConvertXFM(invert_xfm=True), name="invert_registration_matrix")
    skullstrip_functional_10 = Node(fsl.BET(functional=True, mask=True), name="skullstrip_functional_10")
    skullstrip_unwarped_functional = Node(fsl.BET(functional=True), name="skullstrip_unwarped_functional")
    transform_t1 = Node(fsl.FLIRT(apply_xfm=True), name="transform_t1")
    intensity_normalize_t1 = Node(Function(function=IntensityNormalization, input_names=["in_file"], output_names=["out_file"]), name="intensity_normalize_t1")
    transform_t1_to_mni = Node(fsl.FLIRT(dof=12, reference=f"{fsl_dir}/data/standard/MNI152_T1_2mm_brain.nii.gz"), name="transform_t1_to_mni")
    register_t1_to_mni = Node(fsl.FNIRT(config_file="T1_2_MNI152_2mm", ref_file=f"{fsl_dir}/data/standard/MNI152_T1_2mm_brain.nii.gz", field_file=True), name="register_t1_to_mni")

    workflow = Workflow(name="prepare_subject")
    workflow.connect(in_magnitude_image, "out_file", skullstrip_magnitude, "in_file")
    workflow.connect(skullstrip_magnitude, "out_file", erode_magnitude, "in_file")
    workflow.connect(erode_magnitude, "out_file", prepare_fieldmap, "in_magnitude")
    workflow.connect(in_phase_image, "out_file", prepare_fieldmap, "in_phase")
    workflow.connect(prepare_fieldmap, "out_fieldmap", out_field_map, "in_file")
    workflow.connect(in_functional_image, "out_file", median_tf, "in_file")
    workflow.connect(in_functional_image, "out_file", extract_roi_functional, "in_file")
    workflow.connect(in_functional_image, "out_file", extract_roi_functional_10, "in_file")
    workflow.connect(median_tf, "out_value", extract_roi_functional, "t_min")
    workflow.connect(median_tf, "out_value", median_tf_minus_five, "in_value")
    workflow.connect(median_tf_minus_five, "out_value", extract_roi_functional_10, "t_min")
    workflow.connect(extract_roi_functional_10, "roi_file", fugue_correction, "in_file")
    workflow.connect(prepare_fieldmap, "out_fieldmap", fugue_correction, "fmap_in_file")
    workflow.connect(extract_roi_functional, "roi_file", register_functional_to_t1, "epi")
    workflow.connect(in_anatomical_image, "out_file", skullstrip_t1w, "in_file")
    workflow.connect(skullstrip_t1w, "out_file", register_functional_to_t1, "t1_brain")
    workflow.connect(in_anatomical_image, "out_file", register_functional_to_t1, "t1_head")
    workflow.connect(register_functional_to_t1, "epi2str_mat", out_func_to_struct, "in_file")
    workflow.connect(register_functional_to_t1, "epi2str_mat", invert_registration_matrix, "in_file")
    workflow.connect(extract_roi_functional_10, "roi_file", skullstrip_functional_10, "in_file")
    workflow.connect(skullstrip_functional_10, "out_file", out_b0_d, "in_file")
    workflow.connect(skullstrip_functional_10, "mask_file", out_b0_mask, "in_file")
    workflow.connect(fugue_correction, "unwarped_file", skullstrip_unwarped_functional, "in_file")
    workflow.connect(skullstrip_unwarped_functional, "out_file", out_b0_u, "in_file")
    workflow.connect(skullstrip_t1w, "out_file", intensity_normalize_t1, "in_file")
    workflow.connect(intensity_normalize_t1, "out_file", transform_t1, "in_file")
    workflow.connect(skullstrip_unwarped_functional, "out_file", transform_t1, "reference")
    workflow.connect(invert_registration_matrix, "out_file", transform_t1, "in_matrix_file")
    workflow.connect(transform_t1, "out_file", out_t1w, "in_file")
    workflow.connect(intensity_normalize_t1, "out_file", transform_t1_to_mni, "in_file")
    workflow.connect(intensity_normalize_t1, "out_file", register_t1_to_mni, "in_file")
    workflow.connect(transform_t1_to_mni, "out_matrix_file", register_t1_to_mni, "affine_file")
    workflow.connect(register_t1_to_mni, "field_file", out_mni_warp, "in_file")

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
