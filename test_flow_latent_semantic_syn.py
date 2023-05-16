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
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from models.util import get_flow_model
from datasets_prep.coco import CocoImagesAndCaptionsTrain, CocoImagesAndCaptionsValidation
from datasets_prep.ade20k import ADE20kTrain, ADE20kValidation
from datasets_prep.celeb_mask import CelebAMaskTrain, CelebAMaskValidation
import torchvision
from pytorch_fid.fid_score import calculate_fid_given_paths
from diffusers.models import AutoencoderKL
from models.encoder import SpatialRescaler

ADAPTIVE_SOLVER = ["dopri5", "dopri8", "adaptive_heun", "bosh3"]
FIXER_SOLVER = ["euler", "rk4", "midpoint"]


def to_rgb(x):
    x = x.float()
    colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
    x = nn.functional.conv2d(x, weight=colorize)
    x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
    return x

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

class WrapperCondFlow(nn.Module):
    def __init__(self, model, cond):
        super().__init__()
        self.model = model
        self.cond = cond
    
    def forward(self, t, x):
        x = torch.cat([x, self.cond], 1)
        return self.model(t, x)


def sample_and_test(args):
    torch.manual_seed(42)
    device = 'cuda:0'
    
    if args.dataset == "coco":
        dataset = CocoImagesAndCaptionsValidation(size=256, onehot_segmentation=True, use_stuffthing=True)
        num_cls = 182
    elif args.dataset == "ade20k":
        dataset = ADE20kValidation(size=256, crop_size=256, random_crop=False)
        num_cls = 151
    elif args.dataset == "celeba":
        dataset = CelebAMaskValidation(size=256, crop_size=256)
        num_cls = 19
        
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            pin_memory=True)
    args.layout = False
    to_range_0_1 = lambda x: (x + 1.) / 2.

    cond_stage_model = SpatialRescaler(n_stages=3, in_channels=num_cls, out_channels=4, multiplier=0.5).to(device)
    cond_ckpt = torch.load('./saved_info/latent_flow_mask2image/{}/{}/cond_stage_model_{}.pth'.format(args.dataset, args.exp, args.epoch_id), map_location=device)
    for key in list(cond_ckpt.keys()):
        cond_ckpt[key[7:]] = cond_ckpt.pop(key)
    cond_stage_model.load_state_dict(cond_ckpt)
    cond_stage_model.eval()
    
    model =  get_flow_model(args).to(device)
    first_stage_model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    ckpt = torch.load('./saved_info/latent_flow_mask2image/{}/{}/model_{}.pth'.format(args.dataset, args.exp, args.epoch_id), map_location=device)
    #loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    model.load_state_dict(ckpt)
    model.eval()
    print("Finish loading model")

    del ckpt
            
    save_dir = "./mask2image_generated_samples/{}".format(args.dataset)
    # os.makedirs("./testimage_generated_samples/{}".format(args.dataset))
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_cond = WrapperCondFlow(model, cond=None)
    
    # if args.compute_fid:
    for i , (image, segmentation) in enumerate(dataloader):
        segmentation = torch.nn.functional.one_hot(segmentation, num_cls).permute(0, 3, 1, 2)
        seg = segmentation.to(device, non_blocking=True).float()
        with torch.no_grad():
            with torch.no_grad():
                c = cond_stage_model(seg)
            model_cond.cond = c
            z_0 = torch.randn(image.size(0), 4, args.image_size//8, args.image_size//8).to(device)
            fake_sample = sample_from_model(model_cond, z_0, args)[-1]
            fake_image = first_stage_model.decode(fake_sample / args.scale_factor).sample
            fake_image = to_range_0_1(fake_image)
            for j, x in enumerate(fake_image):
                index = i * image.size(0) + j 
                torchvision.utils.save_image(x, './mask2image_generated_samples/{}/{}.jpg'.format(args.dataset, index), normalized=True)
                # torchvision.utils.save_image(to_rgb(seg[j]), './mask2image_generated_samples/{}/mask_{}.jpg'.format(args.dataset, index))
                # torchvision.utils.save_image(image[j], './testimage_generated_samples/{}/gt_{}.jpg'.format(args.dataset, index))
            print('generating batch ', i)
        
        

    
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int,default=425)

    parser.add_argument('--image_size', type=int, default=256,
                            help='size of image')
    parser.add_argument('--num_in_channels', type=int, default=8,
                            help='in channel image')
    parser.add_argument('--num_out_channels', type=int, default=4,
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
    parser.add_argument('--attn_resolutions', nargs='+', type=int, default=(8,4),
                            help='resolution of applying attention')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=(1,2,3,4),
                            help='channel mult')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--num_classes', type=int, default=None,
                            help='num classes')
    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)
    parser.add_argument('--scale_factor', type=float, default=0.18215,
                            help='size of image')
    
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