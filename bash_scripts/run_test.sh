echo "Argument file: $1";
source $1
echo "$(cat $1)"

if [ -z "$CH_MULT" ]; then CH_MULT="1 2 3 4"; fi
if [ -z "$ATTN_RES" ]; then ATTN_RES="16 8 4"; fi
if [ -z "$Bs" ]; then Bs=100; fi
if [ -z "$BASE_CH" ]; then BASE_CH=256; fi

export MASTER_PORT=12004
export PYTHONPATH=$(pwd):$PYTHONPATH

if [[ "${USE_ORIGIN_ADM}" = true ]]; then
    python test_flow_latent.py --exp ${EXP} \
        --dataset ${DATASET} --batch_size ${Bs} --epoch_id ${EPOCH_ID} \
        --image_size ${IMG_SIZE} --f 8 --num_in_channels 4 --num_out_channels 4 \
        --nf ${BASE_CH} --ch_mult ${CH_MULT} --attn_resolution ${ATTN_RES} --num_res_blocks 2 \
        --master_port $MASTER_PORT --num_process_per_node 1 \
        --method ${METHOD} --num_steps ${STEPS} \
        --model_type ${MODEL_TYPE} --use_origin_adm \
        # --measure_time \
        # --use_karras_samplers \
        # --compute_nfe \

else
    python test_flow_latent.py --exp ${EXP} \
        --dataset ${DATASET} --batch_size ${Bs} --epoch_id ${EPOCH_ID} \
        --image_size ${IMG_SIZE} --f 8 --num_in_channels 4 --num_out_channels 4 \
        --nf ${BASE_CH} --ch_mult ${CH_MULT} --attn_resolution ${ATTN_RES} --num_res_blocks 2 \
        --master_port $MASTER_PORT --num_process_per_node 1 \
        --method ${METHOD} --num_steps ${STEPS} \
        --model_type ${MODEL_TYPE} --num_classes 1 --label_dropout 0. \
        # --measure_time \
        # --use_karras_samplers \
        # --compute_nfe \

fi