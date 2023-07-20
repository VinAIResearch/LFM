echo "Argument file: $1";
source $1
echo "$(cat $1)"

if [ -z "$Bs" ]; then Bs=50; fi

export PYTHONPATH=$(pwd):$PYTHONPATH

NUM_GPUS=8
OUTPUT_LOG=${EXP}_${EPOCH_ID}_${METHOD}${STEPS}_cfg${CFG}.log

torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} test_flow_latent_ddp.py --exp $EXP \
    --dataset $DATASET --batch_size ${Bs} --epoch_id $EPOCH_ID \
    --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
    --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
    --model_type $MODEL_TYPE \
    --num_classes 1000 --label_dim 1000 --label_dropout 0.1 \
    --method ${METHOD} --num_steps ${STEPS} \
    --compute_fid --output_log ${OUTPUT_LOG} \
    --cfg_scale ${CFG} \
    # --use_karras_samplers \