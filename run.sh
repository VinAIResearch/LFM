CUDA_VISIBLE_DEVICES=0 python train_flow_latent.py --exp laflo_f8_ \
    --dataset ffhq_256 --datadir data/ffhq/ffhq-lmdb \
    --batch_size 128 --num_epoch 500 \
    --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
    --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
    --lr 2e-4 --scale_factor 0.18215