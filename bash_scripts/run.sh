############################################### ADM ~ CelebA 256 ###############################################
# accelerate launch --num_processes 1 train_flow_latent.py --exp celeb256_f8_adm \
#     --dataset celeba_256 --datadir ../cnf_flow/data/celeba/celeba-lmdb \
#     --batch_size 112 --num_epoch 500 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 2 2 --attn_resolution 16 8 --num_res_blocks 2 \
#     --lr 2e-5 --scale_factor 0.18215 \
#     --save_content --save_content_every 10 \
#     --use_origin_adm


############################################### ADM ~ FFHQ 256 ###############################################
# accelerate launch --num_processes 1 train_flow_latent.py --exp ffhq_f8_adm \
#     --dataset ffhq_256 --datadir data/ffhq/ffhq-lmdb \
#     --batch_size 128 --num_epoch 500 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --lr 2e-5 --scale_factor 0.18215 \
#     --save_content --save_content_every 10 \


############################################### ADM ~ Bed 256 ###############################################
# accelerate launch --num_processes 1 train_flow_latent.py --exp bed_f8_adm \
#     --dataset lsun_bedroom --datadir data/lsun/ \
#     --batch_size 128 --num_epoch 500 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --lr 1e-5 --scale_factor 0.18215 --no_lr_decay \
#     --save_content --save_content_every 10 \


############################################### ADM ~ IMNET 256 ###############################################
# accelerate launch --multi_gpu --num_processes 8 train_flow_latent.py --exp imnet_f8_adm \
#     --dataset imagenet_256 --datadir ./data/imagenet/ --num_classes 1000 \
#     --batch_size 96 --num_epoch 1200 --label_dim 1000 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --lr 1e-4 --scale_factor 0.18215 --no_lr_decay \
#     --save_content --save_content_every 10 \


############################################### DiT-B/2 ~ IMNET 256 ###############################################
# accelerate launch --multi_gpu --num_processes 8 train_flow_latent.py --exp imnet_f8_ditb2 \
#     --dataset imagenet_256 --datadir ./data/imagenet/ \
#     --batch_size 160 --num_epoch 1000 --label_dim 1000 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --lr 1e-4 --scale_factor 0.18215 --no_lr_decay \
#     --model_type DiT-B/2 --num_classes 1000 --label_dropout 0.1 \
#     --save_content --save_content_every 10 \
#     --use_grad_checkpointing \


############################################### DiT-L/2 ~ CelebA 256 ###############################################
# accelerate launch --num_processes 1 train_flow_latent.py --exp celeb_f8_dit \
#     --dataset celeba_256 --datadir data/celeba/celeba-lmdb \
#     --batch_size 32 --num_epoch 500 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --lr 2e-4 --scale_factor 0.18215 --no_lr_decay \
#     --model_type DiT-L/2 --num_classes 1 --label_dropout 0. \
#     --save_content --save_content_every 10 \


############################################### DiT-L/2 ~ FFHQ 256 ###############################################
# accelerate launch --num_processes 1 train_flow_latent.py --exp ffhq_f8_dit \
#     --dataset ffhq_256 --datadir data/ffhq/ffhq-lmdb \
#     --batch_size 32 --num_epoch 500 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --lr 2e-4 --scale_factor 0.18215 \
#     --model_type DiT-L/2 --num_classes 1 --label_dropout 0. \
#     --save_content --save_content_every 10 \


############################################### DiT-L/2 ~ BED 256 ###############################################
# accelerate launch --num_processes 1 train_flow_latent.py --exp bed_f8_dit \
#     --dataset lsun_bedroom --datadir data/lsun/ \
#     --batch_size 32 --num_epoch 600 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --lr 1e-4 --scale_factor 0.18215 --no_lr_decay \
#     --model_type DiT-L/2 --num_classes 1 --label_dropout 0. \
#     --save_content --save_content_every 10 \


############################################### DiT-L/2 ~ Church 256 ###############################################
# accelerate launch --multi_gpu --num_processes 2 train_flow_latent.py --exp church_f8_dit \
#     --dataset lsun_church --datadir data/lsun/ \
#     --batch_size 48 --num_epoch 600 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --lr 1e-4 --scale_factor 0.18215 --no_lr_decay \ 
#     --model_type DiT-L/2 --num_classes 1 --label_dropout 0. \
#     --save_content --save_content_every 10 \


############################################### ADM ~ CelebA 1024 ###############################################
# accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 train_flow_latent.py \
#     --exp celeb1024_f8_adm \
#     --dataset celeba_1024 --datadir data/celeba_1024/celeba-lmdb-1024 \
#     --batch_size 6 --num_epoch 1000 \
#     --image_size 1024 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 1 2 2 4 4 --attn_resolution 16 8 --num_res_blocks 2 \
#     --lr 2e-5 --scale_factor 0.18215 --no_lr_decay \
#     --save_content --save_content_every 10 \


############################################### ADM ~ CelebA 512 ###############################################
# accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 train_flow_latent.py \
#     --exp celeb512_f8_adm \
#     --dataset celeba_512 --datadir data/celeba_512/celeba-lmdb-512 \
#     --batch_size 6 --num_epoch 1000 \
#     --image_size 512 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 2 2 4 --attn_resolution 16 8 --num_res_blocks 2 \
#     --lr 2e-5 --scale_factor 0.18215 --no_lr_decay \
#     --save_content --save_content_every 10 \
#     --use_origin_adm \
