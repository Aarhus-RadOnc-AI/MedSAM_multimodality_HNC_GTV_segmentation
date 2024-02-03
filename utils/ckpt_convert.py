# -*- coding: utf-8 -*-
import torch

# %% convert medsam model checkpoint to sam checkpoint format for convenient inference
sam_ckpt_path = "/home/jintao/gitlab/MedSAM_multimodality_HNC_GTV_segmentation/work_dir/LiteMedSAM/lite_medsam.pth"
medsam_ckpt_path = "/home/jintao/gitlab/MedSAM_multimodality_HNC_GTV_segmentation/work_dir/LiteMedSAM/MedSAM-Lite-hnc-ct-20240131-0209/medsam_lite_best_epoch8.pth"
save_path = "/home/jintao/gitlab/MedSAM_multimodality_HNC_GTV_segmentation/work_dir/LiteMedSAM/MedSAM-Lite-hnc-ct-20240131-0209/medsam_lite_best_convert.pth"
multi_gpu_ckpt = False  # set as True if the model is trained with multi-gpu

medsam_ckpt = torch.load(medsam_ckpt_path)
sam_ckpt = torch.load(sam_ckpt_path)
#medsam_ckpt = torch.load(medsam_ckpt_path)
sam_keys = medsam_ckpt.keys()
print(medsam_ckpt['model'])

#torch.save(sam_ckpt, save_path)
