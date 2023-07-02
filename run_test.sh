#!/bin/sh
#SBATCH --job-name=test_01 # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/haopt12/cnf_flow/slurms/slurm_%A.out # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/haopt12/cnf_flow/slurms/slurm_%A.err # create a error file
#SBATCH --partition=research # choose partition
#SBATCH --gpus-per-node=1
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

export MASTER_PORT=12004
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

MODEL_TYPE=adm
EPOCH_ID=425
DATASET=lsun_bedroom
EXP=laflo_bed_f8_lr5e-5
METHOD=dopri5
STEPS=0
USE_ORIGIN_ADM=False

if [[ ${USE_ORIGIN_ADM} == train ]]; then
    python test_flow_latent.py --exp ${EXP} \
        --dataset ${DATASET} --batch_size 100 --epoch_id ${EPOCH_ID} \
        --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
        --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 --num_res_blocks 2 \
        --use_origin_adm \
        --master_port $MASTER_PORT --num_process_per_node 1 \
        --compute_fid --output_log ${EXP}_${EPOCH_ID}_${METHOD}${STEPS}.log \
        --method dopri5 --num_steps 0 \
        # --measure_time \
        # --use_karras_samplers \
        # --method heun --step_size 50 \
        # --compute_nfe \

else
    python test_flow_latent.py --exp ${EXP} \
        --dataset ${DATASET} --batch_size 100 --epoch_id ${EPOCH_ID} \
        --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
        --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
        --master_port $MASTER_PORT --num_process_per_node 1 \
        --compute_fid --output_log ${EXP}_${EPOCH_ID}_${METHOD}${STEPS}.log \
        --method dopri5 --num_steps 0 \
        # --measure_time \
        # --use_karras_samplers \
        # --method heun --step_size 50 \
        # --compute_nfe \

fi
