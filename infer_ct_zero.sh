#!/bin/bash

dataroot="/mnt/nvme/jintao/medsam_hnc/npz/HNC_CT/tests/"
pred_save_dir="/mnt/nvme/jintao/medsam_hnc/npz/HNC_CT/pred_zero_shot/"
medsam_lite_checkpoint_path="./work_dir/LiteMedSAM/lite_medsam.pth"

nii_img_dir="/data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/imagesTs/"
png_save_dir="/mnt/nvme/jintao/medsam_hnc/png/HNC_CT/pred_zero_shot/"
nii_save_dir="/mnt/nvme/jintao/medsam_hnc/nii/HNC_CT/pred_zero_shot/"



CUDA_VISIBLE_DEVICES=2 python inference_3D.py \
    -data_root ${dataroot} \
    -pred_save_dir ${pred_save_dir} \
    -medsam_lite_checkpoint_path ${medsam_lite_checkpoint_path} \
    -nii_img_dir ${nii_img_dir} \
    -num_workers 8 \
    -png_save_dir ${png_save_dir}  \
    -nii_save_dir ${nii_save_dir}  \
    --save_overlay \
    --overwrite ## overwrite existing predictions, default continue from existing predictions

echo "END TIME: $(date)"
