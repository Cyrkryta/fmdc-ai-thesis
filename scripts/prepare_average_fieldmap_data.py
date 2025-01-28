import json
import os
import tempfile
import time
from os.path import abspath
import shutil
from glob import glob

import numpy as np
import torch
from tqdm import tqdm
from nipype import Node, Workflow
import nipype.interfaces.io as nio
from nipype.interfaces import fsl
from nipype import SelectFiles
import nibabel as nib

from project.data import fmri_data_util


def _create_average_fieldmap_for_subject(subject_path, all_subjects):
    subject_path = os.path.normpath(subject_path)
    registered_fieldmaps = []

    for other_subject in all_subjects:
        with tempfile.TemporaryDirectory() as temp_dir:
            in_anatomical_image = Node(SelectFiles({"out_file": os.path.join(subject_path, "T1w.nii.gz")}), name="in_anatomical_image")
            in_magnitude_image = Node(SelectFiles({"out_file": other_subject['magnitude']}), name="in_magnitude_image")
            in_phase_image = Node(SelectFiles({"out_file": other_subject['phasediff']}), name="in_phase_image")

            out_current_field_map = Node(nio.ExportFile(out_file=abspath(os.path.join(temp_dir, "current_field_map.nii.gz")), clobber=True), name="out_current_field_map")

            erode_magnitude = Node(fsl.maths.ErodeImage(), name="erode_magnitude")
            prepare_fieldmap = Node(fsl.PrepareFieldmap(delta_TE=2.46), name="prepare_fieldmap")
            register_magnitude_to_t1 = Node(fsl.FLIRT(dof=12), name="register_magnitude_to_t1")
            warp_fieldmap_to_t1 = Node(fsl.FLIRT(apply_xfm=True), name="warp_fieldmap_to_t1")

            workflow = Workflow(name="generate_registered_fieldmap")
            workflow.connect(in_magnitude_image, "out_file", erode_magnitude, "in_file")
            workflow.connect(erode_magnitude, "out_file", prepare_fieldmap, "in_magnitude")
            workflow.connect(in_phase_image, "out_file", prepare_fieldmap, "in_phase")
            workflow.connect(in_magnitude_image, "out_file", register_magnitude_to_t1, "in_file")
            workflow.connect(in_anatomical_image, "out_file", register_magnitude_to_t1, "reference")
            workflow.connect(register_magnitude_to_t1, "out_matrix_file", warp_fieldmap_to_t1, "in_matrix_file")
            workflow.connect(prepare_fieldmap, "out_fieldmap", warp_fieldmap_to_t1, "in_file")
            workflow.connect(in_anatomical_image, "out_file", warp_fieldmap_to_t1, "reference")
            workflow.connect(warp_fieldmap_to_t1, "out_file", out_current_field_map, "in_file")

            workflow.run()

            registered_fieldmaps.append(nib.load(os.path.join(temp_dir, 'current_field_map.nii.gz')).get_fdata()[:, :, :, 0])

    original_fieldmap = nib.load(os.path.join(subject_path, 'field_map.nii.gz'))
    fieldmap_mean = np.mean(np.array(registered_fieldmaps), axis=0)
    fieldmap_mean_image = nib.Nifti1Image(fieldmap_mean, original_fieldmap.affine, header=original_fieldmap.header)
    nib.save(fieldmap_mean_image, os.path.join(subject_path, 'mean_field_map.nii.gz'))


def _copy_output_to_new_dir(subject_path, subject_output):
    if not os.path.exists(subject_output):
        os.makedirs(subject_output)

    shutil.copytree(os.path.join(subject_path, "results"), subject_output, dirs_exist_ok=True)


def _convert_all_datasets(dataset_root, original_dataset_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    with open(os.path.join(original_dataset_root, 'preprocessed', 'train_paths.json'), 'r') as f:
        train_paths = json.load(f)['train_paths']
    with open(os.path.join(original_dataset_root, 'training_datasets.json'), 'r') as ds_file:
        dataset_meta = json.load(ds_file)
    train_subjects = []
    for p in train_paths:
        dataset = p.split('/')[-2]
        subject = p.split('/')[-1]
        magnitude_dir = os.path.dirname(dataset_meta[dataset]["paths"]["fmap"]["magnitude"].format(subject=subject, subject_path=os.path.join(original_dataset_root, dataset, subject)))
        train_subjects.append({
            'magnitude': os.path.join(magnitude_dir, 'magnitude1_brain.nii.gz'),
            'phasediff': dataset_meta[dataset]["paths"]["fmap"]["phasediff"].format(subject=subject, subject_path=os.path.join(original_dataset_root, dataset, subject))
        })

    preparation_compute_times = []

    for dataset_dir in glob(os.path.join(dataset_root, 'ds*')):
        dataset_name = dataset_dir.split('/')[-1]

        print(f'Converting dataset "{dataset_name}"...')

        for subject_path in tqdm(glob(f'{dataset_root}/{dataset_name}/sub-*/')):
            subject = os.path.basename(os.path.normpath(subject_path))

            all_files_exist = os.path.isfile(os.path.join(subject_path, 'mean_field_map.nii.gz'))

            if all_files_exist:
                continue
            else:
                print(f'Processing subject {subject}...')
                start = time.time()
                _create_average_fieldmap_for_subject(subject_path, train_subjects)
                end = time.time()
                preparation_compute_times.append(end - start)

    print(f'Mean preparation time per sample in seconds: {np.mean(np.array(preparation_compute_times))}')


def _prepare_output(original_dataset_root, dataset_root, output_root):
    if os.path.isdir(output_root):
        return

    os.makedirs(output_root, exist_ok=True)

    subject_paths = fmri_data_util.collect_all_subject_paths(dataset_paths=glob(dataset_root))

    total_count = len(subject_paths)
    train_count = int(0.7 * total_count)
    val_count = int(0.2 * total_count)
    test_count = total_count - train_count - val_count

    rng = torch.Generator()
    rng.manual_seed(0)
    _, val_paths, _ = torch.utils.data.random_split(
        subject_paths, (train_count, val_count, test_count),
        generator=rng
    )

    for subject_path in list(val_paths):
        dataset = subject_path.split('/')[-2]
        subject = subject_path.split('/')[-1]

        dataset_out_path = os.path.join(output_root, dataset)
        subject_out_path = os.path.join(dataset_out_path, subject)
        os.makedirs(dataset_out_path, exist_ok=True)

        shutil.copy(os.path.join(os.path.dirname(os.path.join(subject_path)), 'dataset_meta.json'), os.path.join(dataset_out_path, 'dataset_meta.json'))
        shutil.copytree(subject_path, subject_out_path)

        with open(os.path.join(original_dataset_root, 'training_datasets.json'), 'r') as ds_file:
            dataset_meta = json.load(ds_file)[dataset]

        magnitude_dir = os.path.dirname(dataset_meta["paths"]["fmap"]["magnitude"].format(subject=subject, subject_path=os.path.join(original_dataset_root, dataset, subject)))
        shutil.copy(os.path.join(magnitude_dir, 'magnitude1_brain.nii.gz'), os.path.join(subject_out_path, 'magnitude.nii.gz'))
        shutil.copy(dataset_meta["paths"]["fmap"]["phasediff"].format(subject=subject, subject_path=os.path.join(original_dataset_root, dataset, subject)), os.path.join(subject_out_path, 'phasediff.nii.gz'))


if __name__ == '__main__':
    """
    Create average fieldmaps from the training dataset. These can then be used as a simple statistical baseline that 
    does not involve any deep learning.
    """

    original_dataset_root = '/Users/jan/Downloads/openneuro-datasets/'
    dataset_root = '/Users/jan/Downloads/openneuro-datasets/preprocessed/ds*/'
    output_root = '/Users/jan/Downloads/openneuro-datasets/preprocessed-average-fieldmaps'

    _prepare_output(original_dataset_root, dataset_root, output_root)
    _convert_all_datasets(dataset_root=output_root, original_dataset_root=original_dataset_root)
