export MASTER_PORT=12004
export PYTHONPATH=$(pwd):$PYTHONPATH

MODEL_TYPE=DiT-B/2
EPOCH_ID=875
DATASET=imagenet_256
EXP=laflo_imnet_f8_ditb2
METHOD=dopri5
STEPS=0
CFG=4.

python test_flow_latent.py --exp ${EXP} \
    --dataset ${DATASET} --batch_size 27 --epoch_id ${EPOCH_ID} \
    --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
    --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 --label_dim 1000 \
    --model_type ${MODEL_TYPE} --num_classes 1000 --label_dropout 0.1 \
    --master_port $MASTER_PORT \
    --cfg_scale ${CFG} \
    --method ${METHOD} --num_steps ${STEPS} \
    # --use_karras_samplers \
    # --measure_time \
    # --compute_nfe \