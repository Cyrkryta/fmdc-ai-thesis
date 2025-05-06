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
import time


original_dataset_root = '/indirect/student/magnuschristensen/dev/fmdc/downloads/original-datasets/test'
template_json_path = '/indirect/student/magnuschristensen/dev/fmdc/downloads/original-datasets/test/test_process_paths.json'
test_processed_root_folder = "/indirect/student/magnuschristensen/dev/fmdc/downloads/processed-datasets/test-processed"
test_processed_dataset_paths = glob(os.path.join(test_processed_root_folder, 'ds*'))

# Load the metadata JSON.
with open(template_json_path, 'r') as f:
    process_templates = json.load(f)
    print(process_templates)

def load_target_subject_image_paths(dataset_id, target_subject_path, process_templates):
    """
    Function for loading target image paths
    """
    templates = process_templates[dataset_id]
    subject = os.path.basename(os.path.normpath(target_subject_path))
    
    # Reconstruct the original subject folder path using the global original_dataset_root.
    original_subject_path = os.path.join(original_dataset_root, dataset_id, subject)
    
    try:
        t1_path = templates["anat"].format(subject_path=original_subject_path, subject=subject)
    except Exception as e:
        print(f"Couldn't format T1 image path: {e}")
        t1_path = None
    
    # For the mean BOLD image, we assume it's already processed and saved in the target subject folder.
    mean_bold_path = os.path.join(target_subject_path, 'b0_d_mean.nii.gz')
    
    return t1_path, mean_bold_path

def load_other_subject_image_paths(dataset_id, subject_path, process_templates):
    """
    Function for loading the original images
    """
    templates = process_templates[dataset_id]
    subject = os.path.basename(os.path.normpath(subject_path))
    original_subject_path = os.path.join(original_dataset_root, dataset_id, subject)
    try:
        anat_path = templates["anat"].format(subject_path=original_subject_path, subject=subject)
        magnitude_path = templates["fmap"]["magnitude"].format(subject_path=original_subject_path, subject=subject)
        phasediff_path = templates["fmap"]["phasediff"].format(subject_path=original_subject_path, subject=subject)
    except Exception as e:
        print(f"Couldn't find the file path: {e}")
        anat_path, magnitude_path, phasediff_path = None, None, None

    mean_bold_path = os.path.join(subject_path, 'b0_d_mean.nii.gz')
    
    return anat_path, magnitude_path, phasediff_path, mean_bold_path



def process_target_subject(target_subject, target_path, other_subjects, dataset_name):
    """
    Function for processing the subjects
    """
    print(f"Processing target: {target_subject}, Path: {target_path}")
    fieldmaps = []
    tgt_anat, tgt_meanb = load_target_subject_image_paths(dataset_name, target_path, process_templates)

    for other_subject, other_path in tqdm(other_subjects.items(), desc="Other subjects"):
            oth_anat, oth_mag, oth_phase, _ = load_other_subject_image_paths(dataset_name, other_path, process_templates)
            tmp_dir = "/indirect/student/magnuschristensen/dev/fmdc/fmdc-ai-thesis/scripts/tmpdir"
            with tempfile.TemporaryDirectory(dir=tmp_dir) as tmpdir:
                wf = Workflow(name=f"warp_{other_subject}_to_{target_subject}", base_dir=tmpdir)

                in_tgt_anat =  Node(SelectFiles({"out_file": tgt_anat}), name="in_tgt_anat")
                in_tgt_meanb = Node(SelectFiles({"out_file": tgt_meanb}), name="in_tgt_meanb")
                in_oth_anat = Node(SelectFiles({"out_file": oth_anat}), name="in_oth_anat")
                in_oth_mag = Node(SelectFiles({"out_file": oth_mag}), name="in_oth_mag")
                in_oth_phase = Node(SelectFiles({"out_file": oth_phase}), name="in_oth_phase")

                strip = Node(fsl.BET(frac=0.5, vertical_gradient=0.0, mask=True), name='strip_mag')
                erode = Node(fsl.maths.ErodeImage(), name='erode_mag')
                prep = Node(fsl.PrepareFieldmap(delta_TE=2.46), name='prep_fmap')
                strip_oth_t1 = Node(fsl.BET(frac=0.5, vertical_gradient=0.0, mask=True), name="oth_t1_strip")
                strip_tgt_t1 = Node(fsl.BET(frac=0.5, vertical_gradient=0.0, mask=True), name="tgt_t1_strip")
                reg1 = Node(fsl.FLIRT(dof=6), name='reg_mag2anat')
                resample = Node(fsl.FLIRT(), name="resample_tgt")
                reg2 = Node(fsl.FLIRT(dof=12), name="reg_anat2tgt")
                concat = Node(fsl.ConvertXFM(concat_xfm=True), name="matrix_concat")
                applyxfm = Node(fsl.FLIRT(apply_xfm=True), name="apply_xfm")
                sink = Node(nio.ExportFile(out_file=abspath(os.path.join(tmpdir, "current_field_map.nii.gz")), clobber=True), name="out_current_field_map")

                # Connect graph
                wf.connect([
                    (in_oth_mag, strip,         [("out_file", "in_file")]),
                    (strip, erode,              [('out_file', 'in_file')]),
                    (in_tgt_anat, strip_tgt_t1, [("out_file", "in_file")]),
                    (in_oth_anat, strip_oth_t1, [("out_file", "in_file")]),
                    (erode, prep,               [('out_file', 'in_magnitude')]),
                    (in_oth_phase, prep,        [("out_file", "in_phase")]),

                    (erode, reg1,               [("out_file", "in_file")]),
                    (strip_oth_t1, reg1,        [("out_file", "reference")]),
                    (strip_tgt_t1, resample,    [("out_file", "in_file")]),
                    (in_tgt_meanb, resample,    [("out_file", "reference")]),
                    (strip_oth_t1, reg2,        [("out_file", "in_file")]),
                    (resample, reg2,            [("out_file", "reference")]),

                    (reg1, concat,              [("out_matrix_file", "in_file")]),
                    (reg2, concat,              [("out_matrix_file", "in_file2")]),

                    (concat, applyxfm,          [("out_file", "in_matrix_file")]),
                    (prep, applyxfm,            [("out_fieldmap", "in_file")]),
                    (resample, applyxfm,        [("out_file", "reference")]),

                    (applyxfm, sink,            [("out_file", "in_file")])
                ])

                wf.run()

                fieldmaps.append(nib.load(os.path.join(tmpdir, 'current_field_map.nii.gz')).get_fdata()[:, :, :, 0])

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
# for dataset_path in tqdm(test_processed_dataset_paths, desc="Datasets"):
all_durations = []
for dataset_path in tqdm(test_processed_dataset_paths, desc="Datasets"):
    # Retrieve the dataset name
    dataset_name = os.path.basename(os.path.normpath(dataset_path))

    # Retrieve the subject paths and create subject dictionary
    subject_paths = sorted(glob(os.path.join(dataset_path, 'sub-*')))
    subjects = {os.path.basename(os.path.normpath(p)): p for p in subject_paths}
    
    # Go through the list of subjects
    # for target_subject, target_path in tqdm(subjects.items(), desc="Target subjects"):
    for target_subject, target_path in tqdm(list(subjects.items()), desc="Target subjects"):
        # Define all other subjects than the target subject
        other_subjects = {name: path for name, path in subjects.items() if name != target_subject}
        t0 = time.time()
        process_target_subject(target_subject, target_path, other_subjects, dataset_name)
        elapsed = time.time() - t0
        print(target_path, target_subject)
        print(other_subjects)
        all_durations.append(elapsed)
        print(f"-> {target_subject} took {elapsed} s")

        # print(f"    For subject {target_subject}, other subjects: {list(other_subjects.keys())}")

if all_durations:
    avg_time = np.mean(all_durations)
    std_time = np.std(all_durations)
    print(f"\nProcessed {len(all_durations)} subjects")
    print(f"All durations: {all_durations}")
    print(f"Average time per subject: {avg_time} s ± {std_time} s")
else:
    print("No subjects processed, so no timing to report.")

"""
VERSION CURRENT BEST (SLOW)
        with tempfile.TemporaryDirectory() as tmp:
            anat_path_other, magnitude_path_other, phasediff_path_other, mean_bold_path_other = load_other_subject_image_paths(dataset_name, other_path, process_templates)
            anat_path_target, mean_bold_path_target = load_target_subject_image_paths(dataset_name, target_path, process_templates)
            os_paths = {"anat": anat_path_other, "mag": magnitude_path_other, "phasediff": phasediff_path_other, "meanBOLD": mean_bold_path_other}
            ts_paths = {"anat": anat_path_target, "meanBOLD": mean_bold_path_target}
            in_target_anat = Node(SelectFiles({"out_file": ts_paths["anat"]}), name="in_target_anat")
            in_target_BOLD_mean = Node(SelectFiles({"out_file": ts_paths["meanBOLD"]}), name="in_target_BOLD_mean")

            in_other_anat = Node(SelectFiles({"out_file": os_paths["anat"]}), name="in_other_anat")
            in_other_mag = Node(SelectFiles({"out_file": os_paths["mag"]}), name="in_other_mag")
            in_other_phasediff = Node(SelectFiles({"out_file": os_paths["phasediff"]}), name="in_other_phasediff")
            in_other_BOLD_mean = Node(SelectFiles({"out_file": os_paths["meanBOLD"]}), name="in_other_BOLD_mean")

            out_current_fieldmap = Node(nio.ExportFile(out_file=abspath(os.path.join(tmp, "current_field_map.nii.gz")), clobber=True), name="out_current_field_map")

            skullstrip_mag = Node(fsl.BET(frac=0.5, vertical_gradient=0.0, mask=True), name="skullstrip_mag")
            erode_mag = Node(fsl.maths.ErodeImage(), name="erode_mag")
            bet_other_anat = Node(fsl.BET(frac=0.5, robust=True), name="bet_other_anat")
            bet_target_anat = Node(fsl.BET(frac=0.5, robust=True),name="bet_target_anat")

            epi_reg_other = Node(fsl.EpiReg(), name="epi_reg_other")
            reg_other_mag_to_MBother = Node(fsl.FLIRT(dof=6), name="reg_other_mag_to_MBother")
            epi_reg_target = Node(fsl.EpiReg(), name="epi_reg_target")
            invert_str2MB_target = Node(fsl.ConvertXFM(invert_xfm=True),name="invert_str2MB_target")
            resample_targetT1_to_MB = Node(fsl.FLIRT(apply_xfm=True), name="resample_targetT1_to_MB")
            reg_other_anat_to_targetT1B = Node(fsl.FLIRT(dof=6), name="reg_other_anat_to_targetT1B")

            prepare_fieldmap = Node(fsl.PrepareFieldmap(delta_TE=2.46), name="prepare_fieldmap")
            concat_mag_to_MB_target_AB = Node(fsl.ConvertXFM(concat_xfm=True), name="concat_mag_to_MB_target_AB")
            concat_mag_to_MB_target_ABC = Node(fsl.ConvertXFM(concat_xfm=True), name="concat_mag_to_MB_target_ABC")

            # invert_epi2str_other = Node(fsl.ConvertXFM(invert_xfm=True), name="invert_epi2str_other")

            # concat_mag_to_MBtarget = Node(fsl.ConvertXFM(concat_xfm=True), name="concat_mag_to_MBtarget")

            warp_fieldmap_to_MBtarget = Node(fsl.FLIRT(apply_xfm=True), name="warp_fieldmap_to_MBtarget")

            wf = Workflow(name="fieldmap_estimation", base_dir=tmp)

            wf.add_nodes([
                in_target_anat, in_target_BOLD_mean,
                in_other_anat, in_other_mag, in_other_phasediff, in_other_BOLD_mean,
                skullstrip_mag, erode_mag,
                bet_other_anat, epi_reg_other, reg_other_mag_to_MBother,
                bet_target_anat, epi_reg_target, invert_str2MB_target,
                resample_targetT1_to_MB, reg_other_anat_to_targetT1B,
                prepare_fieldmap, warp_fieldmap_to_MBtarget, concat_mag_to_MB_target_AB, concat_mag_to_MB_target_ABC
            ])

            wf.connect([
                (in_other_mag, skullstrip_mag, [('out_file', 'in_file')]),
                (skullstrip_mag, erode_mag, [('out_file', 'in_file')]),
                (erode_mag,     reg_other_mag_to_MBother, [('out_file', 'in_file')]),
                (in_other_BOLD_mean, reg_other_mag_to_MBother, [('out_file', 'reference')]),
                (in_other_anat, bet_other_anat, [('out_file', 'in_file')]),
                (in_other_BOLD_mean, epi_reg_other, [('out_file', 'epi')]),
                (in_other_anat,      epi_reg_other, [('out_file', 't1_head')]),
                (bet_other_anat,     epi_reg_other, [('out_file', 't1_brain')]),

                # (epi_reg_other, invert_epi2str_other, [('epi2str_mat', 'in_file')]),

                (in_target_anat, bet_target_anat, [('out_file', 'in_file')]),
                (in_target_BOLD_mean, epi_reg_target, [('out_file', 'epi')]),
                (in_target_anat,      epi_reg_target, [('out_file', 't1_head')]),
                (bet_target_anat,     epi_reg_target, [('out_file', 't1_brain')]),
                (epi_reg_target, invert_str2MB_target, [('epi2str_mat', 'in_file')]),
                (in_target_anat,       resample_targetT1_to_MB, [('out_file', 'in_file')]),
                (in_target_BOLD_mean,  resample_targetT1_to_MB, [('out_file', 'reference')]),
                (invert_str2MB_target, resample_targetT1_to_MB, [('out_file', 'in_matrix_file')]),
                (in_other_anat,        reg_other_anat_to_targetT1B, [('out_file', 'in_file')]),
                (resample_targetT1_to_MB, reg_other_anat_to_targetT1B, [('out_file', 'reference')]),
                (erode_mag,     prepare_fieldmap, [('out_file',   'in_magnitude')]),
                (in_other_phasediff, prepare_fieldmap, [('out_file',   'in_phase')]),
                (reg_other_mag_to_MBother, concat_mag_to_MB_target_AB, [('out_matrix_file', 'in_file')]), # First concat
                # (invert_epi2str_other,     concat_mag_to_MB_target_AB, [('out_file', 'in_file2')]),
                (epi_reg_other, concat_mag_to_MB_target_AB, [('epi2str_mat', 'in_file2')]),
                (concat_mag_to_MB_target_AB, concat_mag_to_MB_target_ABC, [('out_file', 'in_file')]), # Second concat
                (reg_other_anat_to_targetT1B, concat_mag_to_MB_target_ABC, [('out_matrix_file', 'in_file2')]),

                (prepare_fieldmap, warp_fieldmap_to_MBtarget, [('out_fieldmap', 'in_file')]),
                (in_target_BOLD_mean, warp_fieldmap_to_MBtarget, [('out_file', 'reference')]),
                (concat_mag_to_MB_target_ABC, warp_fieldmap_to_MBtarget, [('out_file', 'in_matrix_file')]),
                (warp_fieldmap_to_MBtarget, out_current_fieldmap, [("out_file", "in_file")])
            ])

                # (reg_other_mag_to_MBother, concat_mag_to_MBtarget,[('out_matrix_file', 'in_file')]),
                # (epi_reg_other, concat_mag_to_MBtarget,[('epi2str_mat', 'in_file2')]),
                # (reg_other_anat_to_targetT1B, concat_mag_to_MBtarget, [('out_matrix_file', 'in_file3')]),
                # (concat_mag_to_MBtarget, warp_fieldmap_to_MBtarget, [('out_file', 'in_matrix_file')]),

            wf.run()
"""


    # workflow = Workflow(name="compute_registered_fieldmap")
    
    # workflow.connect([
    #     (in_other_mag, skullstrip_mag, [("out_file", "in_file")]),
    #     (skullstrip_mag, erode_mag, [("out_file", "in_file")]),
    # ])

    # reg_other_t1_to_BOLD = Node(fsl.EpiReg(), name="reg_other_t1_to_BOLD")
    # workflow.connect([
    #     (in_other_anat, reg_other_t1_to_BOLD, [('out_file', 'in_file')]),
    #     (in_other_BOLD_mean, reg_other_t1_to_BOLD, [('out_file', 'epi')]),
    # ])

    # workflow.run()

    # temp_dir = tempfile.mkdtemp()
    # dummy_fieldmap_path = os.path.join(temp_dir, "fieldmap.nii.gz")
    # source_image = os.path.join(target_path, 'b0_d_mean.nii.gz')
    # shutil.copy(source_image, dummy_fieldmap_path)
  


# def estimate_fieldmap(target_path, other_subject_path, dataset_id):
#     with tempfile.TemporaryDirectory() as tmp:
#         anat_path_other, magnitude_path_other, phasediff_path_other, mean_bold_path_other = load_other_subject_image_paths(dataset_id, other_subject_path, process_templates)
#         anat_path_target, mean_bold_path_target = load_target_subject_image_paths(dataset_id, target_path, process_templates)
#         os_paths = {"anat": anat_path_other, "mag": magnitude_path_other, "phasediff": phasediff_path_other, "meanBOLD": mean_bold_path_other}
#         ts_paths = {"anat": anat_path_target, "meanBOLD": mean_bold_path_target}
#         in_target_anat = Node(SelectFiles({"out_file": ts_paths["anat"]}), name="in_target_anat")
#         in_target_BOLD_mean = Node(SelectFiles({"out_file": ts_paths["meanBOLD"]}), name="in_target_BOLD_mean")

#         in_other_anat = Node(SelectFiles({"out_file": os_paths["anat"]}), name="in_other_anat")
#         in_other_mag = Node(SelectFiles({"out_file": os_paths["mag"]}), name="in_other_mag")
#         in_other_phasediff = Node(SelectFiles({"out_file": os_paths["phasediff"]}), name="in_other_phasediff")
#         in_other_BOLD_mean = Node(SelectFiles({"out_file": os_paths["meanBOLD"]}), name="in_other_BOLD_mean")

#         skullstrip_mag = Node(fsl.BET(frac=0.5, vertical_gradient=0.0, mask=True), name="skullstrip_mag")
#         erode_mag = Node(fsl.maths.ErodeImage(), name="erode_mag")
#         bet_other_anat = Node(fsl.BET(frac=0.5, robust=True), name="bet_other_anat")
#         bet_target_anat = Node(fsl.BET(frac=0.5, robust=True),name="bet_target_anat")

#         epi_reg_other = Node(fsl.EpiReg(), name="epi_reg_other")
#         reg_other_mag_to_MBother = Node(fsl.FLIRT(dof=6), name="reg_other_mag_to_MBother")
#         epi_reg_target = Node(fsl.EpiReg(), name="epi_reg_target")
#         invert_str2MB_target = Node(fsl.ConvertXFM(invert_xfm=True),name="invert_str2MB_target")
#         resample_targetT1_to_MB = Node(fsl.FLIRT(apply_xfm=True), name="resample_targetT1_to_MB")
#         reg_other_anat_to_targetT1B = Node(fsl.FLIRT(dof=6), name="reg_other_anat_to_targetT1B")

#         prepare_fieldmap = Node(fsl.PrepareFieldmap(delta_TE=2.46), name="prepare_fieldmap")
#         concat_mag_to_MBtarget = Node(fsl.ConvertXFM(concat_xfm=True), name="concat_mag_to_MBtarget")
#         warp_fieldmap_to_MBtarget = Node(fsl.FLIRT(apply_xfm=True), name="warp_fieldmap_to_MBtarget")

#         wf = Workflow(name="fieldmap_estimation", base_dir=tmp)

#         wf.add_nodes([
#             in_target_anat, in_target_BOLD_mean,
#             in_other_anat, in_other_mag, in_other_phasediff, in_other_BOLD_mean,
#             skullstrip_mag, erode_mag,
#             bet_other_anat, epi_reg_other, reg_other_mag_to_MBother,
#             bet_target_anat, epi_reg_target, invert_str2MB_target,
#             resample_targetT1_to_MB, reg_other_anat_to_targetT1B,
#             prepare_fieldmap, concat_mag_to_MBtarget, warp_fieldmap_to_MBtarget
#         ])

#         wf.connect([
#             (in_other_mag, skullstrip_mag, [('out_file', 'in_file')]),
#             (skullstrip_mag, erode_mag, [('out_file', 'in_file')]),
#             (erode_mag,     reg_other_mag_to_MBother, [('out_file', 'in_file')]),
#             (in_other_BOLD_mean, reg_other_mag_to_MBother, [('out_file', 'reference')]),
#             (in_other_anat, bet_other_anat, [('out_file', 'in_file')]),
#             (in_other_BOLD_mean, epi_reg_other, [('out_file', 'epi')]),
#             (in_other_anat,      epi_reg_other, [('out_file', 't1_head')]),
#             (bet_other_anat,     epi_reg_other, [('out_file', 't1_brain')]),
#             (in_target_anat, bet_target_anat, [('out_file', 'in_file')]),
#             (in_target_BOLD_mean, epi_reg_target, [('out_file', 'epi')]),
#             (in_target_anat,      epi_reg_target, [('out_file', 't1_head')]),
#             (bet_target_anat,     epi_reg_target, [('out_file', 't1_brain')]),
#             (epi_reg_target, invert_str2MB_target, [('epi2str_mat', 'in_file')]),
#             (in_target_anat,       resample_targetT1_to_MB, [('out_file', 'in_file')]),
#             (in_target_BOLD_mean,  resample_targetT1_to_MB, [('out_file', 'reference')]),
#             (invert_str2MB_target, resample_targetT1_to_MB, [('out_file', 'in_matrix_file')]),
#             (in_other_anat,        reg_other_anat_to_targetT1B, [('out_file', 'in_file')]),
#             (resample_targetT1_to_MB, reg_other_anat_to_targetT1B, [('out_file', 'reference')]),
#             (erode_mag,     prepare_fieldmap, [('out_file',   'magnitude_in')]),
#             (in_other_phasediff, prepare_fieldmap, [('out_file',   'phasediff_in')]),
#             (reg_other_mag_to_MBother, concat_mag_to_MBtarget,[('out_matrix_file', 'in_file')]),
#             (epi_reg_other, concat_mag_to_MBtarget,[('epi2str_mat', 'in_file2')]),
#             (reg_other_anat_to_targetT1B, concat_mag_to_MBtarget, [('out_matrix_file', 'in_file3')]),
#             (prepare_fieldmap, warp_fieldmap_to_MBtarget, [('fieldmap_out', 'in_file')]),
#             (in_target_BOLD_mean, warp_fieldmap_to_MBtarget, [('out_file', 'reference')]),
#             (concat_mag_to_MBtarget, warp_fieldmap_to_MBtarget, [('out_file', 'in_matrix_file')]),
#         ])

#         wf.run()

#         return warp_fieldmap_to_MBtarget.result.outputs.out_file