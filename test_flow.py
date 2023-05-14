# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import argparse
import torch
import os
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
from models.util import get_flow_model
import torchvision
from pytorch_fid.fid_score import calculate_fid_given_paths


ADAPTIVE_SOLVER = ["dopri5", "dopri8", "adaptive_heun", "bosh3"]
FIXER_SOLVER = ["euler", "rk4", "midpoint"]

def sample_from_model(model, x_0, args):
    if args.method in ADAPTIVE_SOLVER:
        options = {
            "dtype": torch.float64,
        }
    else:
        options = {
            "step_size": args.step_size,
            "perturb": args.perturb
        }
    if not args.compute_fid:
        model.count_nfe = True
    t = torch.tensor([1., 0.], device="cuda")
    fake_image = odeint(model, 
                        x_0, 
                        t, 
                        method=args.method, 
                        atol = args.atol, 
                        rtol = args.rtol,
                        adjoint_method=args.method,
                        adjoint_atol= args.atol,
                        adjoint_rtol= args.rtol,
                        options=options
                        )
    return fake_image


def sample_and_test(args):
    torch.manual_seed(42)
    device = 'cuda:0'
    
    if args.dataset == 'cifar10':
        real_img_dir = 'pytorch_fid/cifar10_train_stat.npy'
    elif args.dataset == 'celeba_256':
        real_img_dir = 'pytorch_fid/celeba_256_stat.npy'
    elif args.dataset == 'lsun':
        real_img_dir = 'pytorch_fid/lsun_church_stat.npy'
    else:
        real_img_dir = args.real_img_dir
    
    to_range_0_1 = lambda x: (x + 1.) / 2.

    
    model =  get_flow_model(args).to(device)
    ckpt = torch.load('./saved_info/flow_matching/{}/{}/model_{}.pth'.format(args.dataset, args.exp, args.epoch_id), map_location=device)
    print("Finish loading model")
    #loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    model.load_state_dict(ckpt)
    model.eval()
        
    iters_needed = 50000 //args.batch_size
    
    save_dir = "./generated_samples/{}".format(args.dataset)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if args.compute_fid:
        for i in range(iters_needed):
            with torch.no_grad():
                x_0 = torch.randn(args.batch_size, 3, args.image_size, args.image_size).to(device)
                fake_sample = sample_from_model(model, x_0, args)[-1]
                fake_sample = to_range_0_1(fake_sample)
                for j, x in enumerate(fake_sample):
                    index = i * args.batch_size + j 
                    torchvision.utils.save_image(x, './generated_samples/{}/{}.jpg'.format(args.dataset, index))
                print('generating batch ', i)
        
        paths = [save_dir, real_img_dir]
    
        kwargs = {'batch_size': 200, 'device': device, 'dims': 2048}
        fid = calculate_fid_given_paths(paths=paths, **kwargs)
        print('FID = {}'.format(fid))
    else:
        x_0 = torch.randn(args.batch_size, 3,args.image_size, args.image_size).to(device)
        fake_sample = sample_from_model(model, x_0, args)[-1]
        fake_sample = to_range_0_1(fake_sample)
        print("NFE: {}".format(model.nfe))
        torchvision.utils.save_image(fake_sample, './samples_{}_{}_{}_{}_{}.jpg'.format(args.dataset, args.method, args.atol, args.rtol, model.nfe))

    
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int,default=1000)

    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    parser.add_argument('--num_in_channels', type=int, default=3,
                            help='in channel image')
    parser.add_argument('--num_out_channels', type=int, default=3,
                            help='in channel image')
    parser.add_argument('--nf', type=int, default=256,
                            help='channel of image')
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
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--num_classes', type=int, default=None,
                            help='num classes')
    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)
    
    #######################################
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy', help='directory to real images for FID computation')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--num_timesteps', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=200, help='sample generating batch size')
    
    # sampling argument
    parser.add_argument('--atol', type=float, default=1e-5, help='absolute tolerance error')
    parser.add_argument('--rtol', type=float, default=1e-5, help='absolute tolerance error')
    parser.add_argument('--method', type=str, default='dopri5', help='solver_method', choices=["dopri5", "dopri8", "adaptive_heun", "bosh3", "euler", "midpoint", "rk4"])
    parser.add_argument('--step_size', type=float, default=0.01, help='step_size')
    parser.add_argument('--perturb', action='store_true', default=False)
        



   
    args = parser.parse_args()
    
    sample_and_test(args)
    
   
                