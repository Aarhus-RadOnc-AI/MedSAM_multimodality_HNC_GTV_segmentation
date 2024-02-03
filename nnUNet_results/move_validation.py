
import numpy as np
import json 
import os
from os import listdir, makedirs
from os.path import join, isfile, basename
from glob import glob
import shutil
from matplotlib import pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', '--tr_npy_path', type=str,
                        default='/processing/jintao/medsam_hnc/npy/HNC_blend/trains/',
                        help='Path to training npy files; two subfolders: gts and imgs')
    parser.add_argument('-val', '--val_npy_path', type=str,
                        default='/processing/jintao/medsam_hnc/npy/HNC_blend/vals/',
                        help='Path to validation npy files; two subfolders: gts and imgs')
    
    parser.add_argument('-nnunet', '--nnunet_summary_file',type=str, 
                        default='/home/jintao/gitlab/MedSAM_multimodality_HNC_GTV_segmentation/nnUNet_results/val_fold_0/summary.json')

    args = parser.parse_args()

    return args

def extract_patient_ids(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    patient_ids = set()

    for result in data.get("results", {}).get("all", []):
        for key in ["reference", "test"]:
            if key in result:
                # Extract the file name from the path
                file_name = os.path.basename(result[key])
                # Extract the patient ID from the file name
                patient_id = (file_name.split('_')[0] + '_' + file_name.split('_')[1]).split('.nii.gz')[0]
                #print(file_name.split('_')[1])
                patient_ids.add(patient_id)

    return list(patient_ids)

def move_data(args):
    patient_ids = extract_patient_ids(args.nnunet_summary_file)
    print(sorted(patient_ids))
    # Directories
    train_imgs_dir = os.path.join(args.tr_npy_path, 'imgs')
    train_gts_dir = os.path.join(args.tr_npy_path, 'gts')
    val_imgs_dir = os.path.join(args.val_npy_path, 'imgs')
    val_gts_dir = os.path.join(args.val_npy_path, 'gts')

    # Ensure validation directories exist
    os.makedirs(val_imgs_dir, exist_ok=True)
    os.makedirs(val_gts_dir, exist_ok=True)

    # Function to move files based on patient IDs
    def move_files(src_dir, dst_dir):
        for file_name in sorted(os.listdir(src_dir)):
            #print(file_name)
            for patient_id in patient_ids:
                if patient_id in file_name:
                    print(f"Found match: Patient ID {patient_id} in file {file_name}")
                    src_file = os.path.join(src_dir, file_name)
                    dst_file = os.path.join(dst_dir, file_name)
                    shutil.move(src_file, dst_file)
                    break  # Break after finding a match

    # Move files in imgs and gts subdirectories
    move_files(train_imgs_dir, val_imgs_dir)
    move_files(train_gts_dir, val_gts_dir)
    
# %%
if __name__ == "__main__":
    args = get_args()
    move_data(args)
