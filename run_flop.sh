MODEL_TYPE=adm
EPOCH_ID=425
DATASET=lsun_bedroom
EXP=laflo_bed_f8_lr5e-5

CUDA_VISIBLE_DEVICES=0 python compute_flops.py --exp ${EXP} \
    --dataset ${DATASET} --batch_size 1 --epoch_id ${EPOCH_ID} \
    --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
    --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
    # --model_type ${MODEL_TYPE} --num_classes 1 --label_dropout 0. \
