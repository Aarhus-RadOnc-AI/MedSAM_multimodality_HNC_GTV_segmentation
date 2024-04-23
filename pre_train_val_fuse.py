# -*- coding: utf-8 -*-
# %% import packages
# pip install connected-components-3d
import numpy as np

# import nibabel as nib
import SimpleITK as sitk
import os
join = os.path.join
listdir = os.listdir
makedirs = os.makedirs
from tqdm import tqdm
import cc3d
import json 
import multiprocessing as mp
from functools import partial

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-modality_suffix_dict", type=dict, default={'CT': '_0000.nii.gz', 'PET': '_0001.nii.gz', 'T1': '_0002.nii.gz', 'T2': '_0003.nii.gz'}, 
                    help="CT or MR, [default: CT]")
parser.add_argument("-anatomy", type=str, default="HNC_midfuse",
                    help="Anaotmy name, [default: Abd]")
parser.add_argument("-img_name_suffix", type=str, default="_0000.nii.gz",
                    help="Suffix of the image name, [default: _0000.nii.gz]")
parser.add_argument("-gt_name_suffix", type=str, default=".nii.gz",
                    help="Suffix of the ground truth name, [default: .nii.gz]")
parser.add_argument("-img_path", type=str, default="/data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/imagesTr/",
                    help="Path to the nii images, [default: data/FLARE22Train/images]")
parser.add_argument("-gt_path", type=str, default="/data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/labelsTr",
                    help="Path to the ground truth, [default: data/FLARE22Train/labels]")
parser.add_argument("-output_path", type=str, default="/mnt/processing/jintao/medsam_hnc/npy",
                    help="Path to save the npy files, [default: ./data/npy]")
parser.add_argument("-num_workers", type=int, default=4,
                    help="Number of workers, [default: 4]")
parser.add_argument("-window_level", type=int, default=40,
                    help="CT window level, [default: 40]")
parser.add_argument("-window_width", type=int, default=400,
                    help="CT window width, [default: 400]")
parser.add_argument("--save_nii", default=True, action="store_true",
                    help="Save the image and ground truth as nii files for sanity check; they can be removed")
parser.add_argument("--fuse_method", type=str, default="fuse_all_channels",  #fuse_all_channels, repeat_first, combine_rgb, mri_weight_blending_b
                    help="Save the image and ground truth as nii files for sanity check; they can be removed")

args = parser.parse_args()

# convert nii image to npz files, including original image and corresponding masks
#modality = args.modality  # CT or MR
anatomy = args.anatomy  # anantomy + dataset name

img_name_suffix = args.img_name_suffix  # "_0000.nii.gz"
modality_suffix_dict = args.modality_suffix_dict

gt_name_suffix = args.gt_name_suffix  # ".nii.gz"
prefix = anatomy + "_"

nii_path = args.img_path  # path to the nii images
gt_path = args.gt_path  # path to the ground truth
output_path = args.output_path  # path to save the preprocessed files
npy_tr_path = join(output_path, prefix[:-1], "trains")
os.makedirs(npy_tr_path, exist_ok=True)
npy_ts_path = join(output_path, prefix[:-1], "vals")
os.makedirs(npy_ts_path, exist_ok=True)

num_workers = args.num_workers

voxel_num_thre2d = 20
voxel_num_thre3d = 500 # changed from 1000 to 500, some tumors are very small

names = sorted(os.listdir(gt_path))
print(f"ori \# files {len(names)=}")

names = [
    name
    for name in names
    if os.path.exists(join(nii_path, name.split(gt_name_suffix)[0] + list(modality_suffix_dict.values())[0]))
]

print(f"after sanity check \# files {len(names)=}")

# set label ids that are excluded
remove_label_ids = [
]  # remove deodenum since it is scattered in the image, which is hard to specify with the bounding box
tumor_id = 2  # only set this when there are multiple tumors; convert semantic masks to instance masks

# set window level and width
# https://radiopaedia.org/articles/windowing-ct
WINDOW_LEVEL = args.window_level # only for CT images
WINDOW_WIDTH = args.window_width # only for CT images

save_nii = args.save_nii
class ImageReader:
    def __init__(self, img_path, modalities):
        self.img_path = img_path
        self.modalities = modalities
        self.images = {}
        self.sitk_images = {}
        
    def read_images(self, name):
        for modality in self.modalities:
            if modality in modality_suffix_dict:
                img_name = name.split('.nii.gz')[0]  + modality_suffix_dict[modality]
                img_sitk = sitk.ReadImage(os.path.join(self.img_path, img_name))
                self.images[modality] = sitk.GetArrayFromImage(img_sitk)
                self.sitk_images[modality] = img_sitk
                
    def get_image_data(self, modality):
        return self.images.get(modality)  
      
    def get_sitk_data(self, modality):
        return self.sitk_images.get(modality)
    
def fuse_images(images, method='repeat_first'):
    if method == 'repeat_first':
        return np.stack([images[0], images[0], images[0]], axis=-1)
    elif method == 'combine_rgb': 
        return np.stack([images[0], images[1], images[2]], axis=-1)
    elif method == 'mri_weight_blending_b':
        alpha = 0.5  # Weight for the PET image
    
        # Function to rescale the t1 image to the scale of another image
        def rescale_t1(t1, t2):
            #t1 = np.clip(t1, 0, 6)
            #print("start rescale t1,", t1.shape, t2.shape)
            min_t1, max_t1 = t1.min(), t1.max()
            min_other, max_other = t2.min(), t2.max()
            # Rescale the t1 image
            rescaled_t1 = (t1 - min_t1) / (max_t1 - min_t1) * (max_other - min_other) + min_other
            return rescaled_t1
    
        # Rescale t1 image to the range of each of the other images
        # rescaled_pet_for_red = rescale_pet(images[1], images[0])
        # rescaled_pet_for_green = rescale_pet(images[1], images[2])
        rescaled_t1_for_mri = rescale_t1(images[2], images[3])
    
        blended_mri = alpha * rescaled_t1_for_mri + (1 - alpha) * images[3]
    
        # Stacking the blended channels to create the RGB image
        return np.stack([images[0], images[1], blended_mri], axis=-1)
    elif method == 'fuse_all_channels':
        # Determine the maximum number of channels among the input images
        #max_channels = max(image.shape[-1] for image in images)
        max_channels = len(images)
        # Create an empty array to store the fused image
        #fused_image = np.zeros((*images[0].shape[:-1], max_channels))
        fused_image = np.zeros((*images[0].shape, max_channels))
        
        # Iterate over each channel and assign the corresponding image
        for i in range(max_channels):
           # if i < (len(images)-1):
            fused_image[..., i] = images[i]
            
        return fused_image
    else:
        raise ValueError("Unsupported fusion method.")

# %% save preprocessed images and masks as npz files
def preprocess(name, npy_path):
    """
    Preprocess the image and ground truth, and save them as npz files

    Parameters
    ----------
    name : str
        name of the ground truth file
    npy_path : str
        path to save the npz files
    """
    #fuse_method = 'mri_weight_blending_b'
    patient_id = name.split(gt_name_suffix)[0]
    image_name = patient_id + img_name_suffix
    gt_name = name
    gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
    
    gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))
    
    makedirs(join(npy_path, "imgs"), exist_ok=True)
    makedirs(join(npy_path, "gts"), exist_ok=True)

    # remove label ids
    for remove_label_id in remove_label_ids:
        gt_data_ori[gt_data_ori == remove_label_id] = 0
    # label tumor masks as instances and remove from gt_data_ori
    if tumor_id is not None:
        tumor_bw = np.uint8(gt_data_ori == tumor_id)
        gt_data_ori[tumor_bw > 0] = 0
        # label tumor masks as instances
        tumor_inst, tumor_n = cc3d.connected_components(
            tumor_bw, connectivity=26, return_N=True
        )
        # put the tumor instances back to gt_data_ori
        gt_data_ori[tumor_inst > 0] = (
            tumor_inst[tumor_inst > 0] + np.max(gt_data_ori) + 1
        )

    # exclude the objects with less than 1000 pixels in 3D
    gt_data_ori = cc3d.dust(
        gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True
    )
    # remove small objects with less than 100 pixels in 2D slices

    for slice_i in range(gt_data_ori.shape[0]):
        gt_i = gt_data_ori[slice_i, :, :]
        # remove small objects with less than 100 pixels
        # reason: fro such small objects, the main challenge is detection rather than segmentation
        gt_data_ori[slice_i, :, :] = cc3d.dust(
            gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True
        )
    # find non-zero slices
    z_index, _, _ = np.where(gt_data_ori > 0)
    z_index = np.unique(z_index)
    
    reader = ImageReader(nii_path, ['CT', 'PET',  'T1', 'T2'])
    reader.read_images(name)
    
    
    if len(z_index) > 0:
        # crop the ground truth with non-zero slices
        gt_roi = gt_data_ori[z_index, :, :]
        # load image and preprocess
        
        first_img_sitk = sitk.ReadImage(join(nii_path, image_name))
        
        #image_data = sitk.GetArrayFromImage(img_sitk)
        
        # nii preprocess start
        def preprocess_img(image_data, modality):
            if modality == "CT":
                lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
                upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
                image_data_pre = np.clip(image_data, lower_bound, upper_bound)
                image_data_pre = (
                    (image_data_pre - np.min(image_data_pre))
                    / (np.max(image_data_pre) - np.min(image_data_pre))
                    * 255.0
                )
            elif modality == "PET":
                image_data_pre = np.clip(image_data, 0, 6)
                image_data_pre = (
                    (image_data_pre - np.min(image_data_pre))
                    / (np.max(image_data_pre) - np.min(image_data_pre))
                    * 255.0
                )
            else:
                lower_bound, upper_bound = np.percentile(
                    image_data[image_data > 0], 0.5
                ), np.percentile(image_data[image_data > 0], 99.5)
                image_data_pre = np.clip(image_data, lower_bound, upper_bound)
                image_data_pre = image_data
                image_data_pre = (
                    (image_data_pre - np.min(image_data_pre))
                    / (np.max(image_data_pre) - np.min(image_data_pre))
                    * 255.0
                )
                image_data_pre[image_data == 0] = 0
                
            return image_data_pre
        
        img_list = [preprocess_img(reader.get_image_data(modality),modality) for modality in reader.modalities]

        fused_image = fuse_images(img_list, method = args.fuse_method) #fuse_all_channels , 'mri_weight_blending_b'
        #print(fused_image.shape)
        
        fused_image = np.uint8(fused_image)
        img_roi = fused_image[z_index, :, :, :]
        
        for i in range(img_roi.shape[0]):
            img_i = img_roi[i, :, :, :]
            img_3c = img_i # already 3c by fuse images
            #img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)
            
            img_01 = (img_3c - img_3c.min()) / np.clip(
                img_3c.max() - img_3c.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)

            gt_i = gt_roi[i, :, :]

            gt_i = np.uint8(gt_i)
            assert img_01.shape[:2] == gt_i.shape
            np.save(join(npy_path, "imgs", patient_id + "-" + str(i).zfill(3) + ".npy"), img_01)
            np.save(join(npy_path, "gts", patient_id + "-" + str(i).zfill(3) + ".npy"), gt_i)
        
        #np.savez_compressed(join(npz_path, gt_name.split(gt_name_suffix)[0]+'.npz'), imgs=img_roi, gts=gt_roi, spacing=first_img_sitk.GetSpacing())

        # save the image and ground truth as nii files for sanity check;
        # they can be removed
        if save_nii:
            # img_roi_sitk = sitk.GetImageFromArray(img_roi)
            # img_roi_sitk.SetSpacing(first_img_sitk.GetSpacing())
            # sitk.WriteImage(
            #     img_roi_sitk,
            #     join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_img.nii.gz"),
            # )
            gt_roi_sitk = sitk.GetImageFromArray(gt_roi)
            gt_roi_sitk.SetSpacing(first_img_sitk.GetSpacing())
            os.makedirs(join(output_path, prefix[:-1], 'sanity'), exist_ok=True)
            sitk.WriteImage(
                gt_roi_sitk,
                join(output_path, prefix[:-1], 'sanity', gt_name.split(gt_name_suffix)[0] + ".nii.gz"),
            )
            
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
                patient_id = (file_name.split('_')[0] + '_' + file_name.split('_')[1])
                #print(file_name.split('_')[1])
                patient_ids.add(patient_id)

    return list(patient_ids)

if __name__ == "__main__":
    val_patient_ids = extract_patient_ids('/home/jintao/gitlab/MedSAM_multimodality_HNC_GTV_segmentation/nnUNet_results/val_fold_0/summary.json')
    print(sorted(val_patient_ids))
    
    tr_names = [name for name in names if name not in val_patient_ids]
    val_names = [name for name in names if name in val_patient_ids]

    preprocess_tr = partial(preprocess, npy_path=npy_tr_path)
    preprocess_ts = partial(preprocess, npy_path=npy_ts_path)

    with mp.Pool(num_workers) as p: 
        with tqdm(total=len(tr_names)) as pbar:
            pbar.set_description("Preprocessing training data")
            for i, _ in tqdm(enumerate(p.imap_unordered(preprocess_tr, tr_names))):
                pbar.update()
        with tqdm(total=len(val_names)) as pbar:
            pbar.set_description("Preprocessing testing data")
            for i, _ in tqdm(enumerate(p.imap_unordered(preprocess_ts, val_names))):
                pbar.update()
