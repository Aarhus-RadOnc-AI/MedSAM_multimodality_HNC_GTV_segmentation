# %%
import os
import random
import monai
from os import listdir, makedirs
from os.path import join, isfile, basename
from glob import glob
from tqdm import tqdm
from copy import deepcopy
from time import time
from shutil import copyfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import multiprocessing as mp
from torch import distributed as dist
from datetime import datetime

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
import cv2
import torch.nn.functional as F

from matplotlib import pyplot as plt
import argparse

torch.cuda.empty_cache()
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--tr_npy_path', type=str,
                        default='data/npy',
                        help='Path to training npy files; two subfolders: gts and imgs')
    parser.add_argument('-val', '--val_npy_path', type=str,
                        default='data/npy',
                        help='Path to validation npy files; two subfolders: gts and imgs')
    
    parser.add_argument('-task_name', type=str, default='MedSAM-Lite')
    parser.add_argument('-pretrained_checkpoint', type=str, default='lite_medsam.pth',
                        help='Path to pretrained MedSAM-Lite checkpoint')
    parser.add_argument('-work_dir', type=str, default='./work_dir')
    parser.add_argument('--data_aug', action='store_true', default=False,
                        help='use data augmentation during training')
    # train
    parser.add_argument('-num_epochs', type=int, default=1000)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-num_workers', type=int, default=8)
    # Optimizer parameters
    parser.add_argument('-weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    parser.add_argument('-lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (absolute lr)')
    ## Distributed training args
    parser.add_argument('-world_size', type=int, help='world size')
    parser.add_argument('-node_rank', type=int, help='Node rank')
    parser.add_argument('-bucket_cap_mb', type = int, default = 25,
                        help='The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)')
    parser.add_argument('-resume', type = str, default = '', required=False,
                        help="Resuming training from a work_dir")
    parser.add_argument('-init_method', type = str, default = "env://")
    args = parser.parse_args()

    return args


def show_mask(mask, ax, random_color=True):
    #print()
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.45])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.45])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))


@torch.no_grad()
def cal_iou(result, reference):
    
    intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)])
    union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)])
    
    iou = intersection.float() / union.float()
    
    return iou.unsqueeze(1)


def revert_sync_batchnorm(module: torch.nn.Module) -> torch.nn.Module:
    # Code adapted from https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547
    # Original author: Kapil Yedidi (@kapily)
    converted_module = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        # Unfortunately, SyncBatchNorm does not store the original class - if it did
        # we could return the one that was originally created.
        converted_module = nn.BatchNorm2d(
            module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats
        )
        if module.affine:
            with torch.no_grad():
                converted_module.weight = module.weight
                converted_module.bias = module.bias
        converted_module.running_mean = module.running_mean
        converted_module.running_var = module.running_var
        converted_module.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            converted_module.qconfig = module.qconfig
    for name, child in module.named_children():
        converted_module.add_module(name, revert_sync_batchnorm(child))
    del module

    return converted_module


class NpyDataset(Dataset): 
    def __init__(self, data_root, image_size=256, bbox_shift=10, data_aug=True):
        self.data_root = data_root
        self.gt_path = join(data_root, 'gts')
        self.img_path = join(data_root, 'imgs')
        self.gt_path_files = sorted(glob(join(self.gt_path, '*.npy'), recursive=True))
        self.gt_path_files = [file for file in self.gt_path_files if isfile(join(self.img_path, basename(file)))]
        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug
    
    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = basename(self.gt_path_files[index])
        assert img_name == basename(self.gt_path_files[index]), 'img gt name error' + self.gt_path_files[index] + self.npy_files[index]

        img_3c = np.load(join(self.img_path, img_name), 'r', allow_pickle=True)  # (H, W, C)

        # Resizing and normalization
        img_resize = self.resize_longest_side(img_3c)
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None)  # normalize to [0, 1], (H, W, C)
        img_padded = self.pad_image(img_resize)  # (256, 256, C)

        # convert the shape to (C, H, W)
        img_padded = np.transpose(img_padded, (2, 0, 1))  # (C, 256, 256)
        assert np.max(img_padded) <= 1.0 and np.min(img_padded) >= 0.0, 'image should be normalized to [0, 1]'

        gt = np.load(self.gt_path_files[index], 'r', allow_pickle=True) # multiple labels [0, 1,4,5...], (256,256)
        assert gt.max() >= 1, 'gt should have at least one label'
        gt = cv2.resize(
            gt,
            (img_resize.shape[1], img_resize.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
        gt = self.pad_image(gt) # (256, 256)
        label_ids = np.unique(gt)[1:]
        try:
            gt2D = np.uint8(gt == random.choice(label_ids.tolist())) # only one label, (256, 256)
        except:
            print(img_name, 'label_ids.tolist()', label_ids.tolist())
            gt2D = np.uint8(gt == np.max(gt)) # only one label, (256, 256)
        # add data augmentation: random fliplr and random flipud
        if self.data_aug:
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
                # print('DA with flip left right')
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
                # print('DA with flip upside down')
        gt2D = np.uint8(gt2D > 0)
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return {
            "image": torch.tensor(img_padded).float(),
            "gt2D": torch.tensor(gt2D[None, :,:]).long(),
            "bboxes": torch.tensor(bboxes[None, None, ...]).float(), # (B, 1, 4)
            "image_name": img_name,
            "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
            "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long()
        }
class NchannelNpyDataset(Dataset): 
    def __init__(self, data_root, image_size=256, bbox_shift=10, data_aug=True):
        self.data_root = data_root
        self.gt_path = join(data_root, 'gts')
        self.img_path = join(data_root, 'imgs')
        self.gt_path_files = sorted(glob(join(self.gt_path, '*.npy'), recursive=True))
        self.gt_path_files = [file for file in self.gt_path_files if isfile(join(self.img_path, basename(file)))]
        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug
    
    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = basename(self.gt_path_files[index])
        assert img_name == basename(self.gt_path_files[index]), 'img gt name error' + self.gt_path_files[index] + self.npy_files[index]

        # Load image, assumed shape (H, W, C) with C=4
        img_4c = np.load(join(self.img_path, img_name), allow_pickle=True)
        # Load and process GT as in your original code
        gt = np.load(self.gt_path_files[index], allow_pickle=True)
        
        # Process each of the 4 channels separately and repeat them to form 3-channel images
        img_processed_list = []
        
        #get a random number to check if need to augment data
        random_value = 0
        
        for c in range(img_4c.shape[2]):
            single_channel_img = img_4c[:, :, c]

            # Repeat the single channel to form a 3-channel image
            img_3c = np.repeat(single_channel_img[:, :, np.newaxis], 3, axis=2)

            # Your existing preprocessing (resizing, normalization, padding)
            img_resize = self.resize_longest_side(img_3c)  # Adjust your method to handle 3-channel input
            img_normalize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None)
            
            #processing groud truth label using the first image preprocess
            if c == 0:
                gt = cv2.resize(gt, (img_normalize.shape[1], img_normalize.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                gt = self.pad_image(gt)
                label_ids = np.unique(gt)[1:]
                try:
                    gt2D = np.uint8(gt == random.choice(label_ids.tolist()))
                except:
                    print(img_name, 'label_ids.tolist()', label_ids.tolist())
                    gt2D = np.uint8(gt == np.max(gt))
            
            # pad image
            img_padded = self.pad_image(img_normalize)
            # Convert shape to (3, H, W) 
            img_padded = np.transpose(img_padded, (2, 0, 1))
            assert np.max(img_padded) <= 1.0 and np.min(img_padded) >= 0.0, 'image should be normalized to [0, 1]'
            
            # add data augmentation: random fliplr and random flipud
            if self.data_aug:
                if random_value > 0.5:
                    img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
                    gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
                    # print('DA with flip left right')
                if random_value > 0.5:
                    img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
                    gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
                    # print('DA with flip upside down')

            # and append to list
            img_processed_list.append(img_padded)
        img_processed_list  = np.array(img_processed_list)
        #print("img_processed_list>>>>", img_processed_list.shape)
        img_processed_list = np.transpose(img_processed_list, (1,2,3,0))
        #print("img_processed_list<<<<", img_processed_list.shape)
        
        # Combine processed images into a single tensor with dimensions (3, H, W, N)
        img_tensor = torch.tensor(img_processed_list).float()  # Adds batch dimension

        # make bbox based on ground truth 
        gt2D = np.uint8(gt2D > 0)
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        
        return {
            "image": img_tensor,
            "gt2D": torch.tensor(gt2D[None, :,:]).long(),
            "bboxes": torch.tensor(bboxes[None, None, ...]).float(), # (B, 1, 4)
            "image_name": img_name,
            "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
            "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long()
        }
    def resize_longest_side(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        long_side_length = self.target_length
        oldh, oldw = image.shape[0], image.shape[1]
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww, newh = int(neww + 0.5), int(newh + 0.5)
        target_size = (neww, newh)

        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    def pad_image(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        # Pad
        h, w = image.shape[0], image.shape[1]
        padh = self.image_size - h
        padw = self.image_size - w
        if len(image.shape) == 3: ## Pad image
            image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
        else: ## Pad gt mask
            image_padded = np.pad(image, ((0, padh), (0, padw)))

        return image_padded

def collate_fn(batch):
    """
    Collate function for PyTorch DataLoader.
    """
    batch_dict = {}
    for key in batch[0].keys():
        if key == "image_name":
            batch_dict[key] = [sample[key] for sample in batch]
        else:
            batch_dict[key] = torch.stack([sample[key] for sample in batch], dim=0)

    return batch_dict

#%% sanity test of dataset class
def sanity_check_dataset(args):
    print('tr_npy_path', args.tr_npy_path)
    tr_dataset = NchannelNpyDataset(args.tr_npy_path, data_aug=args.data_aug)
    print('len(tr_dataset)', len(tr_dataset))
    tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    makedirs(args.work_dir, exist_ok=True)
    for step, batch in enumerate(tr_dataloader):
        # print(image.shape, gt.shape, bboxes.shape)
        # show the example
        _, axs = plt.subplots(1, 2, figsize=(10, 10))
        idx = random.randint(0, 4)

        images = batch["image"]
        gt = batch["gt2D"]
        bboxes = batch["bboxes"]
        names_temp = batch["image_name"]

        for i in range(images.shape[-1]):
            image = images[..., i]
           # print("image>>>>>>",image.shape)
            
            axs[0].imshow(image[idx].cpu().permute(1,2,0).numpy())
            show_mask(gt[idx].cpu().squeeze().numpy(), axs[0])
            show_box(bboxes[idx].numpy().squeeze(), axs[0])
            axs[0].axis('off')
            # set title
            axs[0].set_title(names_temp[idx])
            idx = random.randint(4, 7)
            axs[1].imshow(image[idx].cpu().permute(1,2,0).numpy())
            show_mask(gt[idx].cpu().squeeze().numpy(), axs[1])
            show_box(bboxes[idx].numpy().squeeze(), axs[1])
            axs[1].axis('off')
            # set title
            axs[1].set_title(names_temp[idx])
            # plt.show()  
        
            plt.subplots_adjust(wspace=0.01, hspace=0)
            plt.savefig(
                join(args.work_dir, f'medsam_lite-train_bbox_prompt_sanitycheck_DA_{i}.png'),
                bbox_inches='tight',
                dpi=300
            )
        plt.close()
        break

# %%
class MedSAM_Lite(nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder
        ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, boxes):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        low_res_logits, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        ) # (B, 1, 256, 256)

        return low_res_logits, iou_predictions

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing
        """
        # Crop
        masks = masks[:, :, :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks
#updated MedSAM model with capability to fuse embeddings at middle 
class MedSAM_Lite_Midfuse(nn.Module):
    def __init__(self,
                 image_encoder,
                 mask_decoder,
                 prompt_encoder
                 ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
    def forward(self, images, boxes):
        # Reshape the tensor to combine modalities and batch dimensions
        B, C, H, W, N = images.size()
        images = images.view(-1, C, H, W)

        # Feed the reshaped tensor to the encoder
        image_embedding = self.image_encoder(images)  # (B*N, 256, 64, 64)

        # Reshape the output embedding back to the original shape
        #image_embedding = image_embedding.view(B, N, -1, image_embedding.size(-2), image_embedding.size(-1))  # (B, N, 256, 64, 64)
        image_embedding = image_embedding.view(B, N, 256, image_embedding.size(-2), image_embedding.size(-1))  # (B, N, 256, 64, 64)

        # Fuse the embeddings along the modality dimension
        fused_image_embedding = image_embedding.mean(dim=1)  

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        low_res_logits, iou_predictions = self.mask_decoder(
            image_embeddings=fused_image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )  # (B, 1, 256, 256)
        return low_res_logits, iou_predictions
    

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing
        """
        # Crop
        masks = masks[:, :, :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks
    
def main(args):
    ngpus_per_node = torch.cuda.device_count()
    print("Spwaning processces")
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    node_rank = int(args.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    world_size = args.world_size
    print(f"[Rank {rank}]: Use GPU: {gpu} for training")
    is_main_host = rank == 0
    if is_main_host:
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
        model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
        makedirs(model_save_path, exist_ok=True)
        copyfile(
            __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
        )
    torch.cuda.set_device(gpu)
    device = torch.device("cuda:{}".format(gpu))
    dist.init_process_group(
        backend="nccl", init_method=args.init_method, rank=rank, world_size=world_size
    )

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    medsam_lite_image_encoder = TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[
            64, ## (64, 256, 256)
            128, ## (128, 128, 128)
            160, ## (160, 64, 64)
            320 ## (320, 64, 64) 
        ],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8
    )

    medsam_lite_prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16
    )

    medsam_lite_mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
    )
    # medsam_lite_model = MedSAM_Lite(
    #     image_encoder = medsam_lite_image_encoder,
    #     mask_decoder = medsam_lite_mask_decoder,
    #     prompt_encoder = medsam_lite_prompt_encoder
    # )
        
    medsam_lite_model_mid_fuse = MedSAM_Lite_Midfuse(
        image_encoder = medsam_lite_image_encoder,
        mask_decoder = medsam_lite_mask_decoder,
        prompt_encoder = medsam_lite_prompt_encoder
    )
    if (not os.path.exists(args.resume)) and isfile(args.pretrained_checkpoint):
        ## Load pretrained checkpoint if there's no checkpoint to resume from and there's a pretrained checkpoint
        print(f"Loading pretrained checkpoint from {args.pretrained_checkpoint}")
        medsam_lite_checkpoint = torch.load(args.pretrained_checkpoint, map_location="cpu")
        medsam_lite_model_mid_fuse.load_state_dict(medsam_lite_checkpoint, strict=True)

    medsam_lite_model_mid_fuse = medsam_lite_model_mid_fuse.to(device)

    # # updated to load encoder weights multiple times. 
    # if (not os.path.exists(args.resume)) and isfile(args.pretrained_checkpoint):
    #     print(f"Loading pretrained checkpoint from {args.pretrained_checkpoint}")
    #     #Load the pretrained checkpoint of the original MedSAM model:
    #     medsam_lite_checkpoint = torch.load(args.pretrained_checkpoint, map_location="cpu")
    #     medsam_lite_state_dict = medsam_lite_checkpoint["model_state_dict"]
        
    #     # medsam_lite_mid_fuse_state_dict = {}
    #     # for key, value in medsam_lite_state_dict.items():
    #     #     if key.startswith("image_encoder."):
    #     #         for i in range(len(medsam_lite_model_mid_fuse.image_encoders)):
    #     #             medsam_lite_mid_fuse_state_dict[f"image_encoders.{i}.{key[14:]}"] = value
    #     #     else:
    #     #         medsam_lite_mid_fuse_state_dict[key] = value
        
    #     medsam_lite_model_mid_fuse.load_state_dict(medsam_lite_state_dict, strict=False)

    #     medsam_lite_model_mid_fuse = medsam_lite_model_mid_fuse.to(device)



    ## Make sure there's only 2d BN layers, so that I can revert them properly
    for module in medsam_lite_model_mid_fuse.modules():
        cls_name = module.__class__.__name__
        if "BatchNorm" in cls_name:
            assert cls_name == "BatchNorm2d" 
    medsam_lite_model_mid_fuse = nn.SyncBatchNorm.convert_sync_batchnorm(medsam_lite_model_mid_fuse)

    medsam_lite_model_mid_fuse = nn.parallel.DistributedDataParallel(
        medsam_lite_model_mid_fuse,
        device_ids=[gpu],
        output_device=gpu,
        find_unused_parameters=True,
        bucket_cap_mb=args.bucket_cap_mb
    )
    medsam_lite_model_mid_fuse.train()
    # %%
    print(f"MedSAM Lite size: {sum(p.numel() for p in medsam_lite_model_mid_fuse.parameters())}")
    # %%
    optimizer = optim.AdamW(
        medsam_lite_model_mid_fuse.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.9,
        patience=5,
        cooldown=0
    )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    ce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    iou_loss = nn.MSELoss(reduction='mean')
    # %%
    data_root = args.tr_npy_path
    data_val = args.val_npy_path
    
    train_dataset = NchannelNpyDataset(data_root=data_root, data_aug=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    val_dataset = NchannelNpyDataset(data_root=data_val, data_aug=False)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=collate_fn
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=collate_fn
    )
    # %%

    if os.path.exists(args.resume):
        ckpt_folders = sorted(listdir(args.resume))
        ckpt_folders = [f for f in ckpt_folders if (f.startswith(args.task_name) and isfile(join(args.resume, f, 'medsam_lite_latest.pth')))]
        print('*'*20)
        print('existing ckpts in', args.resume, ckpt_folders)
        # find the latest ckpt folders
        time_strings = [f.split(args.task_name + '-')[-1] for f in ckpt_folders]
        dates = [datetime.strptime(f, '%Y%m%d-%H%M') for f in time_strings]
        print(dates)
        latest_date = max(dates)
        latest_ckpt = join(args.work_dir, args.task_name + '-' + latest_date.strftime('%Y%m%d-%H%M'), 'medsam_lite_latest.pth')
        print('Loading from', latest_ckpt)
        checkpoint = torch.load(latest_ckpt, map_location=device)
        medsam_lite_model_mid_fuse.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["loss"]
        print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_loss = 1e10

    train_losses = []
    val_losses = []
    epoch_times = []
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = [1e10 for _ in range(len(train_loader))]
        
        epoch_start_time = time()
        pbar = tqdm(train_loader)
        for step, batch in enumerate(pbar):
            image = batch["image"]
            gt2D = batch["gt2D"]
            boxes = batch["bboxes"]
            optimizer.zero_grad()
            image, gt2D, boxes = image.to(device), gt2D.to(device), boxes.to(device)
            logits_pred, iou_pred = medsam_lite_model_mid_fuse(image, boxes)
            l_seg = seg_loss(logits_pred, gt2D)
            l_ce = ce_loss(logits_pred, gt2D.float())
            mask_loss = l_seg + l_ce
            with torch.no_grad():
                iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
            l_iou = iou_loss(iou_pred, iou_gt)
            loss = mask_loss + l_iou
            epoch_loss[step] = loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"[RANK {rank}] Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")

        epoch_end_time = time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        epoch_loss_world = [None for _ in range(world_size)]
        dist.all_gather_object(epoch_loss_world, epoch_loss)
        
        epoch_loss_reduced = np.vstack(epoch_loss_world).mean()
        train_losses.append(epoch_loss_reduced)
        lr_scheduler.step(epoch_loss_reduced)
        
        
        epoch_val_loss = [1e10 for _ in range(len(val_loader))]
        pbar_val = tqdm(val_loader)
        with torch.no_grad():
            for step, batch in enumerate(pbar_val):
            
                image = batch["image"]
                gt2D = batch["gt2D"]
                boxes = batch["bboxes"]
                image, gt2D, boxes = image.to(device), gt2D.to(device), boxes.to(device)
                logits_pred, iou_pred = medsam_lite_model_mid_fuse(image, boxes)
                l_seg = seg_loss(logits_pred, gt2D)
                #l_ce = ce_loss(logits_pred, gt2D.float())
                #mask_loss = l_seg + l_ce
                #iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
                #l_iou = iou_loss(iou_pred, iou_gt)
                loss = l_seg #mask_loss + l_iou
                epoch_val_loss[step] = loss.item()
                pbar_val.set_description(f"[RANK {rank}] Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, val_loss: {loss.item():.4f}")
                
            epoch_val_loss_world = [None for _ in range(world_size)]
            dist.all_gather_object(epoch_val_loss_world, epoch_val_loss)
            epoch_val_loss_reduced = np.vstack(epoch_val_loss_world).mean()
            val_losses.append(epoch_val_loss_reduced)
        #dist.barrier()
        
        if is_main_host:
            module_revert_sync_BN = revert_sync_batchnorm(deepcopy(medsam_lite_model_mid_fuse.module))
            weights = module_revert_sync_BN.state_dict()
            checkpoint = {
                "model": weights,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "loss": epoch_loss_reduced,
                "val_loss": epoch_val_loss_reduced,
                "best_loss": best_loss,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_lite_latest.pth"))
            
        if epoch_val_loss_reduced < best_loss:
            print(f"New best loss: {best_loss:.4f} -> {epoch_val_loss_reduced:.4f}")
            best_loss = epoch_val_loss_reduced
            if is_main_host:
                checkpoint["best_loss"] = best_loss
                torch.save(checkpoint, join(model_save_path, "medsam_lite_best.pth"))
                
        
        dist.barrier()
        epoch_loss_reduced = 1e10
        epoch_val_loss_reduced = 1e10
        # %% plot loss
        if is_main_host:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            axes[0].title.set_text("Dice + Binary Cross Entropy + IoU Loss")
            axes[0].plot(train_losses, label='Train Loss')
            axes[0].plot(val_losses, label='Validation Loss')
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[1].plot(epoch_times)
            axes[1].title.set_text("Epoch Duration")
            axes[1].set_ylabel("Duration (s)")
            axes[1].set_xlabel("Epoch")
            plt.tight_layout()
            plt.savefig(join(model_save_path, "log.png"))
            plt.close()
        dist.barrier()

        
# %%
if __name__ == "__main__":
    args = get_args()
    sanity_check_dataset(args)
    main(args)
