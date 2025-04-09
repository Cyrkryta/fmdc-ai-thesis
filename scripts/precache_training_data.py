import os
import glob
import argparse
from project.data.fmri_data_util import load_data_from_path_for_train
from project.data.fmri_data_util import collect_all_subject_paths

def precache_all_subjects(subject_paths):
    """
    Function for precaching the subjects
    """
    for subject_path in subject_paths:
        try:
            load_data_from_path_for_train(subject_path=subject_path, use_cache=True)
        except Exception as e:
            print(f"Error processing {subject_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, help="Root directory containing ds* folders")
    args = parser.parse_args()
    subject_paths = glob.glob(os.path.join(args.dataset_root, "ds*", "sub-*"))
    print(f"Found {len(subject_paths)} subjects")
    precache_all_subjects(subject_paths)