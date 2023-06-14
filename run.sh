#!/bin/sh
#SBATCH --job-name=bedadm # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/haopt12/cnf_flow/slurms/slurm_%A.out # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/haopt12/cnf_flow/slurms/slurm_%A.err # create a error file
#SBATCH --partition=research # choose partition
#SBATCH --gpus-per-node=5
#SBATCH --cpus-per-task=32 # 80
#SBATCH --mem-per-gpu=32GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10-00:00          # total run time limit (DD-HH:MM)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail          # send email when job fails
#SBATCH --mail-user=v.haopt12@vinai.io

set -x
set -e

export MASTER_PORT=10008
export WORLD_SIZE=1

export SLURM_JOB_NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')
export SLURM_NODELIST=$SLURM_JOB_NODELIST
master_address=$(echo $SLURM_JOB_NODELIST | cut -d' ' -f1)
export MASTER_ADDRESS=$master_address

echo MASTER_ADDRESS=${MASTER_ADDRESS}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${WORLD_SIZE}
echo "NODELIST="${SLURM_NODELIST}

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

export PYTHONPATH=$(pwd):$PYTHONPATH

############################################### ADM ~ CelebA 256 ###############################################
# CUDA_VISIBLE_DEVICES=1 python train_flow_latent.py --exp laflo_celeba_f8_lr2e-5_bs32 \
#     --dataset celeba_256 --datadir data/celeba/celeba-lmdb \
#     --batch_size 32 --num_epoch 500 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --lr 2e-5 --scale_factor 0.18215 \
#     --save_content --save_content_every 10 \
#     --master_port $MASTER_PORT


############################################### ADM ~ FFHQ 256 ###############################################
# CUDA_VISIBLE_DEVICES=0 python train_flow_latent.py --exp laflo_f8_lr2e-5 \
#     --dataset ffhq_256 --datadir data/ffhq/ffhq-lmdb \
#     --batch_size 128 --num_epoch 500 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --lr 2e-5 --scale_factor 0.18215 \
#     --save_content_every 10 \
#     --master_port $MASTER_PORT


############################################### ADM ~ Bed 256 ###############################################
# python train_flow_latent.py --exp laflo_bed_f8_lr5e-5 \
#     --dataset lsun_bedroom --datadir data/lsun/ \
#     --batch_size 128 --num_epoch 800 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --lr 1e-5 --scale_factor 0.18215 --no_lr_decay \
#     --save_content --save_content_every 10 \
#     --model_ckpt model_500.pth \
#     --master_port $MASTER_PORT --num_process_per_node 1 \


############################################### ADM ~ IMNET 256 ###############################################
# python train_flow_latent.py --exp laflo_imnet_f8_pixel \
#     --dataset imagenet_256 --datadir ./data/imagenet/ --num_classes 1000 \
#     --batch_size 96 --num_epoch 1800 --label_dim 1000 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --lr 1e-4 --scale_factor 0.18215 --no_lr_decay \
#     --save_content --save_content_every 10 \
#     --master_port $MASTER_PORT --num_process_per_node 8 --resume \

############################################### DiT-B/2 ~ IMNET 256 ###############################################
# python train_flow_latent.py --exp laflo_imnet_f8_ditb2 \
#     --dataset imagenet_256 --datadir ./data/imagenet/ \
#     --batch_size 144 --num_epoch 1800 --label_dim 1000 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --lr 1e-4 --scale_factor 0.18215 --no_lr_decay \
#     --model_type DiT-B/2 --num_classes 1000 --label_dropout 0.1 \
#     --save_content --save_content_every 10 \
#     --master_port $MASTER_PORT --num_process_per_node 8 \
#     --use_bf16 --use_grad_checkpointing \

accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 train_flow_latent_faster.py --exp laflo_imnet_f8_ditb2 \
    --dataset imagenet_256 --datadir ./data/imagenet/ \
    --batch_size 160 --num_epoch 1800 --label_dim 1000 \
    --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
    --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
    --lr 1e-4 --scale_factor 0.18215 --no_lr_decay \
    --model_type DiT-B/2 --num_classes 1000 --label_dropout 0.1 \
    --save_content --save_content_every 10 \
    --use_grad_checkpointing \


############################################### DiT-L/2 ~ CelebA 256 ###############################################
# CUDA_VISIBLE_DEVICES=1 python train_flow_latent.py --exp laflo_celeb_f8_dit \
#     --dataset celeba_256 --datadir data/celeba/celeba-lmdb \
#     --batch_size 32 --num_epoch 800 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --lr 1.0000346977833825e-05 --scale_factor 0.18215 --no_lr_decay \
#     --model_type DiT-L/2 --num_classes 1 --label_dropout 0. \
#     --save_content --save_content_every 10 \
#     --model_ckpt model_500.pth \
#     --master_port $MASTER_PORT


############################################### DiT-L/4 ~ CelebA 1024 ###############################################
# python train_flow_latent.py --exp laflo_celeb_f8_ditl4 \
#     --dataset celeba_1024 --datadir data/celeba_1024/celeba-lmdb-1024 \
#     --batch_size 32 --num_epoch 500 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --lr 2e-4 --scale_factor 0.18215 --no_lr_decay \
#     --model_type DiT-L/4 --num_classes 1 --label_dropout 0. \
#     --save_content --save_content_every 10 \
#     --master_port $MASTER_PORT \
#     # --model_ckpt model_500.pth \


############################################### DiT-L/2 ~ FFHQ 256 ###############################################
# CUDA_VISIBLE_DEVICES=0 python train_flow_latent.py --exp laflo_ffhq_f8_dit \
#     --dataset ffhq_256 --datadir data/ffhq/ffhq-lmdb \
#     --batch_size 32 --num_epoch 500 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --lr 2e-4 --scale_factor 0.18215 \
#     --model_type DiT-L/2 --num_classes 1 --label_dropout 0. \
#     --save_content_every 10 \
#     --master_port $MASTER_PORT

############################################### DiT-L/2 ~ BED 256 ###############################################
# python train_flow_latent.py --exp laflo_bed_f8_dit \
#     --dataset lsun_bedroom --datadir data/lsun/ \
#     --batch_size 32 --num_epoch 800 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --lr 6.863658713347484e-05 --scale_factor 0.18215 \
#     --model_type DiT-L/2 --num_classes 1 --label_dropout 0. \
#     --save_content --save_content_every 10 \
#     --no_lr_decay \
#     --resume \
#     --master_port $MASTER_PORT --num_process_per_node 1 \
#     # --model_ckpt model_300.pth \


############################################### DiT-L/2 ~ Church 256 ###############################################
# python train_flow_latent.py --exp laflo_church_f8_dit \
#     --dataset lsun_church --datadir data/lsun/ \
#     --batch_size 48 --num_epoch 800 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --lr 6.863658713347484e-05 --scale_factor 0.18215 \
#     --model_type DiT-L/2 --num_classes 1 --label_dropout 0. \
#     --model_ckpt model_500.pth \
#     --no_lr_decay \
#     --save_content_every 10 \
#     --master_port $MASTER_PORT --num_process_per_node 2
