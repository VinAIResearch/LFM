echo "Argument file: $1";
source $1
echo "$(cat $1)"

export PYTHONPATH=$(pwd):$PYTHONPATH

NUM_GPUS=8

if [[ ${USE_ORIGIN_ADM} == True ]]; then
    torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} test_flow_latent_ddp.py --exp $EXP \
        --dataset $DATASET --batch_size 100 --epoch_id $EPOCH_ID \
        --image_size ${IMG_SIZE} --f 8 --num_in_channels 4 --num_out_channels 4 \
        --nf 256 --ch_mult ${CH_MULT} --attn_resolution ${ATTN_RES} --num_res_blocks 2 \
        --model_type ${MODEL_TYPE} \
        --method ${METHOD} --num_steps ${STEPS} \
        --compute_fid --output_log $OUTPUT_LOG \
        --num_classes 1 --label_dropout 0. \
        --use_origin_adm \
        # --use_karras_samplers \
else
    torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} test_flow_latent_ddp.py --exp $EXP \
        --dataset $DATASET --batch_size 100 --epoch_id $EPOCH_ID \
        --image_size ${IMG_SIZE} --f 8 --num_in_channels 4 --num_out_channels 4 \
        --nf 256 --ch_mult ${CH_MULT} --attn_resolution ${ATTN_RES} --num_res_blocks 2 \
        --model_type ${MODEL_TYPE} \
        --method ${METHOD} --num_steps ${STEPS} \
        --compute_fid --output_log $OUTPUT_LOG \
        --num_classes 1 --label_dropout 0. \
        # --use_karras_samplers \

fi

