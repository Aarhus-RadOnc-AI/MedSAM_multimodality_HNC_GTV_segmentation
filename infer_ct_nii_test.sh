#!/bin/bash

nii_gts_dir="/mnt/data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/labelsTs/"
nii_img_dir="/mnt/data/jintao/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task901_AUH/imagesTs/"

medsam_lite_checkpoint_path="./work_dir/LiteMedSAM/lite_medsam.pth"

png_save_dir="/processing/jintao/medsam_hnc/png/HNC_blend/pred_zero_shot/"
nii_pred_dir="/processing/jintao/medsam_hnc/nii/HNC_blend/pred_zero_shot/"
nii_box_dir="/processing/jintao/medsam_hnc/nii/HNC_blend/pred_zero_shot_box/"

CUDA_VISIBLE_DEVICES=0 python inference_3D_finetune_nii.py \
    -nii_gts_dir ${nii_gts_dir} \
    -medsam_lite_checkpoint_path ${medsam_lite_checkpoint_path} \
    -nii_img_dir ${nii_img_dir} \
    -num_workers 8 \
    -png_save_dir ${png_save_dir}  \
    -nii_pred_dir ${nii_pred_dir}  \
    -nii_box_dir ${nii_box_dir}  \
    --save_overlay \
    --overwrite \
    --zero_shot
    
echo "END TIME: $(date)"
