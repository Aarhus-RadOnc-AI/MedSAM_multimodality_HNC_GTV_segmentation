#!/bin/bash


GPUS_PER_NODE=2 # <- Specify the number of GPUs per machine here

# ## Master node setup
MAIN_HOST=`hostname -s`
export MASTER_ADDR=$MAIN_HOST

# # Get a free port using python
export MASTER_PORT=$(python - <<EOF
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('', 0))  # OS will allocate a free port
free_port = sock.getsockname()[1]
sock.close()
print(free_port)
EOF
)

export NNODES=1
NNODESODE_RANK=0
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES)) # M nodes x N GPUs

echo "nnodes: ${NNODES}"

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1

export NCCL_DEBUG=INFO

dataroot="/processing/jintao/medsam_hnc/npy/HNC_midfuse/trains"
dataval="/processing/jintao/medsam_hnc/npy/HNC_midfuse/vals"
pretrained_checkpoint="./work_dir/LiteMedSAM/lite_medsam.pth"
resume="/home/jintao/gitlab/MedSAM_multimodality_HNC_GTV_segmentation/work_dir/LiteMedSAM/"
CUDA_VISIBLE_DEVICES=0,1 python train_midfuse_multi_gpus_with_val.py \
    -i ${dataroot} \
    -val ${dataval} \
    -task_name MedSAM-Lite-hnc-midfuse \
    -pretrained_checkpoint ${pretrained_checkpoint} \
    -work_dir ./work_dir/LiteMedSAM \
    -num_epochs 12 \
    -batch_size 4 \
    -num_workers 4 \
    -lr 0.0003 \
    --data_aug \
    -world_size ${WORLD_SIZE} \
    -node_rank ${NODE_RANK} \
    -init_method tcp://${MASTER_ADDR}:${MASTER_PORT} 
    -resume ${resume}

echo "END TIME: $(date)"
