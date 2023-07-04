MODEL_TYPE=DiT-B/2
EPOCH_ID=875
DATASET=imagenet_256
EXP=laflo_imnet_f8_ditb2
METHOD=dopri5
STEPS=0
CFG=1.
NUM_GPUS=8

CUDA_VISIBLE_DEVICES={device} torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} test_flow_latent_ddp.py --exp $EXP \
    --dataset $DATASET --batch_size 50 --epoch_id $EPOCH_ID \
    --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
    --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
    --model_type $MODEL_TYPE \
    --num_classes 1000 --label_dim 1000 --label_dropout 0.1 \
    --method ${METHOD} --num_steps ${STEPS} \
    --compute_fid --output_log ${EXP}_${EPOCH_ID}_${METHOD}${STEPS}.log \
    --cfg_scale {cfg_scale} \
    # --use_karras_samplers \