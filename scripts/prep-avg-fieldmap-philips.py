
import os
from glob import glob
import numpy as np
import nibabel as nib
# import tempfile
from tqdm import tqdm
import json
import shutil
from nipype import Node, Workflow
from nipype.interfaces.io import SelectFiles
from nipype.interfaces import fsl
import nipype.interfaces.io as nio
from os.path import abspath
import tempfile

original_dataset_root = '/indirect/student/magnuschristensen/dev/fmdc/downloads/philips/philips-original'
template_json_path = "/indirect/student/magnuschristensen/dev/fmdc/downloads/philips/philips-original/philips-config.json"
test_processed_root_folder = "/indirect/student/magnuschristensen/dev/fmdc/downloads/philips/philips-processed"
test_processed_dataset_paths = glob(os.path.join(test_processed_root_folder, "ds*"))

# Load the metadata JSON.
with open(template_json_path, 'r') as f:
    process_templates = json.load(f)
    print(process_templates)

def load_target_subject_image_paths(dataset_id, target_subject_path, process_templates):
    """
    Load T1w image and bold image (to retrieve the mean bold)
    """
    templates = process_templates[dataset_id]
    subject = os.path.basename(os.path.normpath(target_subject_path))
    original_subject_path = os.path.join(original_dataset_root, dataset_id, subject)
    processed_subject_path = os.path.join(test_processed_root_folder, dataset_id, subject)
    t1_path = templates["anat"].format(subject_path=original_subject_path, subject=subject)
    bold_path = os.path.join(processed_subject_path, "b0_d.nii.gz")
   
    return t1_path, bold_path

def load_other_subject_image_paths(dataset_id, subject_path, process_templates):
    """
    Load anat, mag, phase for other subject.
    """
    templates = process_templates[dataset_id]
    subject = os.path.basename(os.path.normpath(subject_path))
    original_subject_path = os.path.join(original_dataset_root, dataset_id, subject)
    anat_path = templates["anat"].format(subject_path=original_subject_path, subject=subject)
    magnitude_path = templates["fmap"]["magnitude"].format(subject_path=original_subject_path, subject=subject)
    phasediff_path = templates["fmap"]["phasediff"].format(subject_path=original_subject_path, subject=subject)    
    return anat_path, magnitude_path, phasediff_path



def process_target_subject(target_subject, target_path, other_subjects, dataset_name):
    """
    Process a target subject:
      - Compute mean BOLD from b0_d
      - Prepare Philips fieldmap via FUGUE
      - Register each other subject's fieldmap into target B0 space
      - Average and save mean_fieldmap
    """

    print(f"Processing target: {target_subject}, path: {target_path}")
    tgt_anat, tgt_b0d = load_target_subject_image_paths(dataset_name, target_path, process_templates)
    fieldmaps = []
    for other_subject, other_path in tqdm(other_subjects.items(), desc="Other Philips subjects"):
        oth_anat, oth_mag, oth_phase = load_other_subject_image_paths(dataset_name, other_path, process_templates)
        with tempfile.TemporaryDirectory() as tmpdir:
            wf = Workflow(name=f"warp_{other_subject}_to_{target_subject}", base_dir=tmpdir)
            in_tgt_anat =  Node(SelectFiles({"out_file": tgt_anat}), name="in_tgt_anat")
            in_tgt_b0d = Node(SelectFiles({"out_file": tgt_b0d}), name="in_tgt_meanb")
            in_oth_anat = Node(SelectFiles({"out_file": oth_anat}), name="in_oth_anat")
            in_oth_mag = Node(SelectFiles({"out_file": oth_mag}), name="in_oth_mag")
            in_oth_phase = Node(SelectFiles({"out_file": oth_phase}), name="in_oth_phase")

            delta_TE = process_templates[dataset_name]["ET2"] - process_templates[dataset_name]["ET1"]

            mean_bold = Node(fsl.maths.MeanImage(dimension='T'), name='mean_bold')
            strip = Node(fsl.BET(frac=0.5, vertical_gradient=0.0, mask=True), name='strip_mag')
            deg2rad = Node(fsl.maths.MathsCommand(args="-mul 3.141592653589793 -div 180"), name="deg2rad")
            rad2rads = Node(fsl.maths.MathsCommand(args=f"-div {delta_TE}"), name="rad2rads") # Scale radians to rad/s
            mask_fmap = Node(fsl.ApplyMask(), name="mask_fmap")
            fugue_prep = Node(fsl.FUGUE(smooth3d=1, despike_2dfilter=True, median_2dfilter=True, save_fmap=True), name="reg_fmap") # Derived fieldmap
            strip_oth_t1 = Node(fsl.BET(frac=0.5, vertical_gradient=0.0, mask=True), name="oth_t1_strip")
            strip_tgt_t1 = Node(fsl.BET(frac=0.5, vertical_gradient=0.0, mask=True), name="tgt_t1_strip")
            reg1 = Node(fsl.FLIRT(dof=6), name='reg_mag2anat')
            resample = Node(fsl.FLIRT(), name="resample_tgt")
            reg2 = Node(fsl.FLIRT(dof=12), name="reg_anat2tgt")
            concat = Node(fsl.ConvertXFM(concat_xfm=True), name="matrix_concat")
            applyxfm = Node(fsl.FLIRT(apply_xfm=True), name="apply_xfm")
            sink = Node(nio.ExportFile(out_file=abspath(os.path.join(tmpdir, "current_field_map.nii.gz")), clobber=True), name="out_current_field_map")


            wf.connect([
                (in_tgt_b0d, mean_bold, [("out_file", "in_file")]),
                (in_oth_mag, strip, [("out_file", "in_file")]),
                (in_oth_phase, deg2rad, [("out_file", "in_file")]),
                (deg2rad, rad2rads, [("out_file", "in_file")]),
                (rad2rads, mask_fmap, [("out_file", "in_file")]),
                (strip, mask_fmap, [("out_file", "mask_file")]),
                (mask_fmap, fugue_prep, [("out_file", "fmap_in_file")]),
                (in_tgt_anat, strip_tgt_t1, [("out_file", "in_file")]),
                (in_oth_anat, strip_oth_t1, [("out_file", "in_file")]),
                (strip, reg1, [("out_file", "in_file")]),
                (strip_oth_t1, reg1, [("out_file", "reference")]),
                (strip_tgt_t1, resample, [("out_file", "in_file")]),
                (mean_bold, resample, [("out_file", "reference")]),
                (strip_oth_t1, reg2, [("out_file", "in_file")]),
                (resample, reg2, [("out_file", "reference")]),
                (reg1, concat, [("out_matrix_file", "in_file")]),
                (reg2, concat, [("out_matrix_file", "in_file2")]),
                (concat, applyxfm, [("out_file", "in_matrix_file")]),
                (fugue_prep, applyxfm, [("fmap_out_file", "in_file")]),
                (resample, applyxfm, [("out_file", "reference")]),
                (applyxfm, sink, [("out_file", "in_file")])
            ])

            wf.run()

            fieldmaps.append(nib.load(os.path.join(tmpdir, 'current_field_map.nii.gz')).get_fdata())

    if not fieldmaps:
        print("No fieldmaps created")
        return
   
    orig_fieldmap = nib.load(os.path.join(target_path, "field_map.nii.gz"))
    avg_fieldmap = np.mean(np.array(fieldmaps), axis=0)
    avg_fieldmap_img = nib.Nifti1Image(avg_fieldmap, orig_fieldmap.affine, header=orig_fieldmap.header)
    out_path = os.path.join(target_path, 'mean_fieldmap.nii.gz')
    nib.save(avg_fieldmap_img, out_path)
    print(f"Saved average fieldmap to {out_path}\n")

# Go through each of the datasets
for dataset_path in tqdm(test_processed_dataset_paths, desc="Datasets"):
    # Retrieve the dataset name
    dataset_name = os.path.basename(os.path.normpath(dataset_path))

    # Retrieve the subject paths and create subject dictionary
    subject_paths = sorted(glob(os.path.join(dataset_path, 'sub-*')))
    subjects = {os.path.basename(os.path.normpath(p)): p for p in subject_paths}
   
    # Go through the list of subjects
    # first_item = first_item = list(subjects.items())

    for target_subject, target_path in tqdm(list(subjects.items()), desc="Target subjects"):
        other_subjects = {name: path for name, path in subjects.items() if name != target_subject}
        process_target_subject(target_subject, target_path, other_subjects, dataset_name)
        print(target_path, target_subject)
        print(other_subjects)