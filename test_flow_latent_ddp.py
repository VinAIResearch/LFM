# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import os
import argparse
import numpy as np
from tqdm import tqdm
import math
from functools import partial

import torch
from torch import nn
import torchvision
from PIL import Image
from torchdiffeq import odeint_adjoint as ozdeint
from diffusers.models import AutoencoderKL

import torch.distributed as dist
from torch.multiprocessing import Process

from models import create_network

from pytorch_fid.fid_score import calculate_fid_given_paths
from ddp_utils import init_processes
from sampler.karras_sample import karras_sample
from sampler.random_util import get_generator

from test_flow_latent import ADAPTIVE_SOLVER, FIXER_SOLVER
from test_flow_latent import sample_from_model, sample_from_model2



def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True  # True: fast but may lead to some small numerical differences
    torch.set_grad_enabled(False)
    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    # seed = args.seed * dist.get_world_size() + rank
    seed = args.seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.dataset == 'cifar10':
        real_img_dir = 'pytorch_fid/cifar10_train_stat.npy'
    elif args.dataset == 'celeba_256':
        real_img_dir = 'pytorch_fid/celebahq_stat.npy'
    elif args.dataset == 'lsun_church':
        real_img_dir = 'pytorch_fid/lsun_church_stat.npy'
    elif args.dataset == "ffhq_256":
        real_img_dir = 'pytorch_fid/ffhq_stat.npy'
    elif args.dataset == "lsun_bedroom":
        real_img_dir = 'pytorch_fid/lsun_bedroom_stat.npy'
    elif args.dataset in ["latent_imagenet_256", "imagenet_256"]:
        real_img_dir = 'pytorch_fid/imagenet_stat.npy'
    else:
        real_img_dir = args.real_img_dir

    to_range_0_1 = lambda x: (x + 1.) / 2.

    model = create_network(args).to(device)
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

    first_stage_model = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device)

    ckpt = torch.load('./saved_info/latent_flow/{}/{}/model_{}.pth'.format(args.dataset, args.exp, args.epoch_id))
    print("Finish loading model")
    #loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    del ckpt

    save_dir = "./generated_samples/{}/exp{}_ep{}_m{}".format(args.dataset, args.exp, args.epoch_id, args.method)
    # save_dir = "./generated_samples/{}".format(args.dataset)
    if args.method in FIXER_SOLVER:
        save_dir += "_s{}".format(args.num_steps)
    if args.cfg_scale > 1.:
        save_dir += "_cfg{}".format(args.cfg_scale)

    if rank == 0 and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # seed generator
    #### seed should be aligned with rank
    generator = get_generator(args.generator, args.n_sample, seed)

    def run_sampling(num_samples, generator):
        x = generator.randn(num_samples, 4, args.image_size//8, args.image_size//8).to(device)
        if args.num_classes in [None, 1]:
            model_kwargs = {}
        else:
            y = generator.randint(0, args.num_classes, (num_samples,), device=device)
            # Setup classifier-free guidance:
            if args.cfg_scale > 1.:
                x = torch.cat([x, x], 0)
                y_null = torch.tensor([args.num_classes] * num_samples, device=device) if "DiT" in args.model_type else torch.zeros_like(y)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            else:
                model_kwargs = dict(y=y)

        if not args.use_karras_samplers:
            fake_sample = sample_from_model(model, x, model_kwargs, args)[-1]
        else:
            fake_sample = sample_from_model2(model, x, model_kwargs, generator, args)

        if args.cfg_scale > 1.:
            fake_sample, _ = fake_sample.chunk(2, dim=0)  # Remove null class samples

        fake_image = first_stage_model.decode(fake_sample / args.scale_factor).sample
        return fake_image

    print("Compute fid")
    dist.barrier()
    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.n_sample / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    iters_needed = int(samples_needed_this_gpu // n)
    pbar = range(iters_needed)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    for i in pbar:
        with torch.no_grad():
            fake_image = run_sampling(args.batch_size, generator)
            fake_image = (torch.clamp(to_range_0_1(fake_image), 0, 1) * 255.).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            for j, x in enumerate(fake_image):
                index = j * dist.get_world_size() + rank + total
                Image.fromarray(x).save(f"{save_dir}/{index}.jpg")
                # torchvision.utils.save_image(x, '{}/{}.jpg'.format(save_dir, index))
            if rank == 0:
                print('generating batch ', i)
            total += global_batch_size

    # make sure all processes have finished
    dist.barrier()
    if rank == 0:
        paths = [save_dir, real_img_dir]
        kwargs = {'batch_size': 200, 'device': device, 'dims': 2048}
        fid = calculate_fid_given_paths(paths=paths, **kwargs)
        print('FID = {}'.format(fid))
        with open(args.output_log, "a") as f:
            f.write('Epoch = {}, FID = {}, cfg_scale = {}\n'.format(args.epoch_id, fid, args.cfg_scale))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('flow-matching parameters')
    parser.add_argument('--generator', type=str, default="determ",
                        help='type of seed generator', choices=["dummy", "determ", "determ-indiv"])
    parser.add_argument('--seed', type=int, default=42,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--use_origin_adm', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--compute_nfe', action='store_true', default=False,
                            help='whether or not compute NFE')
    parser.add_argument('--measure_time', action='store_true', default=False,
                            help='wheter or not measure time')
    parser.add_argument('--epoch_id', type=int,default=1000)

    parser.add_argument('--model_type', type=str, default="adm",
                            help='model_type', choices=['adm', 'ncsn++', 'ddpm++', 'DiT-B/2', 'DiT-L/2', 'DiT-XL/2'])
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    parser.add_argument('--f', type=int, default=8,
                            help='downsample rate of input image by the autoencoder')
    parser.add_argument('--scale_factor', type=float, default=0.18215,
                            help='size of image')
    parser.add_argument('--num_in_channels', type=int, default=3,
                            help='in channel image')
    parser.add_argument('--num_out_channels', type=int, default=3,
                            help='in channel image')
    parser.add_argument('--nf', type=int, default=256,
                            help='channel of image')
    parser.add_argument('--n_sample', type=int, default=50000,
                            help='number of sampled images')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument("--resamp_with_conv", type=bool, default=True)
    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--num_heads', type=int, default=4,
                            help='number of head')
    parser.add_argument('--num_head_upsample', type=int, default=-1,
                            help='number of head upsample')
    parser.add_argument('--num_head_channels', type=int, default=-1,
                            help='number of head channels')
    parser.add_argument('--attn_resolutions', nargs='+', type=int, default=(16,),
                            help='resolution of applying attention')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=(1,2,2,2),
                            help='channel mult')
    parser.add_argument('--label_dim', type=int, default=0,
                            help='label dimension, 0 if unconditional')
    parser.add_argument('--augment_dim', type=int, default=0,
                            help='dimension of augmented label, 0 if not used')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--num_classes', type=int, default=None,
                            help='num classes')
    parser.add_argument('--label_dropout', type=float, default=0.,
                            help='Dropout probability of class labels for classifier-free guidance')
    parser.add_argument('--cfg_scale', type=float, default=1.,
                            help='Scale for classifier-free guidance')

    # parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    # parser.add_argument("--resblock_updown", type=bool, default=False)
    # parser.add_argument("--use_new_attention_order", type=bool, default=False)

    parser.add_argument('--pretrained_autoencoder_ckpt', type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument('--output_log', type=str, default="")

    #######################################
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy', help='directory to real images for FID computation')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--num_steps', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=200, help='sample generating batch size')

    # sampling argument
    parser.add_argument('--use_karras_samplers', action='store_true', default=False)
    parser.add_argument('--atol', type=float, default=1e-5, help='absolute tolerance error')
    parser.add_argument('--rtol', type=float, default=1e-5, help='absolute tolerance error')
    parser.add_argument('--method', type=str, default='dopri5', help='solver_method', choices=["dopri5", "dopri8", "adaptive_heun", "bosh3",
        "euler", "midpoint", "rk4", "heun", "multistep", "stochastic", "dpm"])
    parser.add_argument('--step_size', type=float, default=0.01, help='step_size')
    parser.add_argument('--perturb', action='store_true', default=False)

    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--master_port', type=str, default='6000',
                        help='port for master')

    args = parser.parse_args()
    main(args)
