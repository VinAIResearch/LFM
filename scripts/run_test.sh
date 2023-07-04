echo "Argument file: $1";
source $1
echo "$(cat $1)"

export MASTER_PORT=12004
export PYTHONPATH=$(pwd):$PYTHONPATH

if [[ ${USE_ORIGIN_ADM} == True ]]; then
    python test_flow_latent.py --exp ${EXP} \
        --dataset ${DATASET} --batch_size 100 --epoch_id ${EPOCH_ID} \
        --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
        --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 --num_res_blocks 2 \
        --master_port $MASTER_PORT --num_process_per_node 1 \
        --method ${METHOD} --num_steps ${STEPS} \
        --use_origin_adm \
        # --measure_time \
        # --use_karras_samplers \
        # --compute_nfe \

else
    python test_flow_latent.py --exp ${EXP} \
        --dataset ${DATASET} --batch_size 100 --epoch_id ${EPOCH_ID} \
        --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
        --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
        --master_port $MASTER_PORT --num_process_per_node 1 \
        --method dopri5 --num_steps ${STEPS} \
        # --measure_time \
        # --use_karras_samplers \
        # --compute_nfe \

fi