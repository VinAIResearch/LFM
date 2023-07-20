echo "Argument file: $1";
source $1
echo "$(cat $1)"

if [ -z "$Bs" ]; then Bs=50; fi

export MASTER_PORT=12004
export PYTHONPATH=$(pwd):$PYTHONPATH

python test_flow_latent.py --exp ${EXP} \
    --dataset ${DATASET} --batch_size ${Bs} --epoch_id ${EPOCH_ID} \
    --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
    --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 --label_dim 1000 \
    --model_type ${MODEL_TYPE} --num_classes 1000 --label_dropout 0.1 \
    --master_port $MASTER_PORT \
    --method ${METHOD} --num_steps ${STEPS} \
    --cfg_scale ${CFG} \
    # --use_karras_samplers \
    # --measure_time \
    # --compute_nfe \