# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from diffusers.models import AutoencoderKL
from models import get_flow_model
from PIL import Image
from torch.utils.data import Dataset
from torchdiffeq import odeint_adjoint as odeint


ADAPTIVE_SOLVER = ["dopri5", "dopri8", "adaptive_heun", "bosh3"]
FIXER_SOLVER = ["euler", "rk4", "midpoint"]


class CustomizedInpaintingEvalDataset(Dataset):
    def __init__(self, indir, inmasked, transform=None):
        self.indir = indir
        self.inmasked = inmasked
        self.transform = transform
        self.iter_i = 0

    def __len__(self):
        return 2993

    def __getitem__(self, item):
        img = np.array(Image.open(os.path.join(self.indir, f"{item:06d}.jpg")).convert("RGB"))
        img = np.transpose(img, (2, 0, 1))

        mask = np.array(Image.open(os.path.join(self.inmasked, f"{item:06d}.png")))
        mask = torch.from_numpy(mask)
        mask = mask / 255.0
        mask = 1 - mask

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        masked_img = (1 - mask) * img
        if self.transform is not None:
            img = self.transform(img)
        self.iter_i += 1
        img = img * 2.0 - 1.0
        mask = mask * 2.0 - 1.0
        masked_img = masked_img * 2.0 - 1.0
        mask = mask.unsqueeze(0)
        return img, mask, masked_img


def sample_from_model(model, x_0, args):
    if args.method in ADAPTIVE_SOLVER:
        options = {
            "dtype": torch.float64,
        }
    else:
        options = {"step_size": args.step_size, "perturb": args.perturb}
    if not args.compute_fid:
        model.count_nfe = True
    t = torch.tensor([1.0, 0.0], device="cuda")
    fake_image = odeint(
        model,
        x_0,
        t,
        method=args.method,
        atol=args.atol,
        rtol=args.rtol,
        adjoint_method=args.method,
        adjoint_atol=args.atol,
        adjoint_rtol=args.rtol,
        options=options,
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
    device = "cuda:0"

    to_range_0_1 = lambda x: (x + 1.0) / 2.0

    dataset = CustomizedInpaintingEvalDataset("./gt_celeb/", "./dataset/masks_val_256_small_eval")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    img, mask, masked_img = next(iter(dataloader))

    # torchvision.utils.save_image(mask[0:4], "imask.png", normalize = True)
    # torchvision.utils.save_image(img[0:4], "image.png", normalize=True)
    # torchvision.utils.save_image(masked_img[0:4], "image_m.png", normalize=True)
    # os.makedirs("mask_celeb", exist_ok=True)
    # global_step = 0
    # for iteration, (img, mask, masked_img) in enumerate(dataloader):
    #     for i in range(img.size(0)):
    #         torchvision.utils.save_image(masked_img[i], os.path.join("mask_celeb", 'imask_{}.png'.format(global_step)), normalize=True)
    #         global_step += 1
    # # exit()

    args.layout = False

    model = get_flow_model(args).to(device)
    first_stage_model = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device)
    ckpt = torch.load(
        "./saved_info/latent_flow_inpaint/{}/{}/model_{}.pth".format(args.dataset, args.exp, args.epoch_id),
        map_location=device,
    )
    print("Finish loading model")
    # loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    model.load_state_dict(ckpt)
    model.eval()

    del ckpt

    save_dir = "./inpainting_generated_samples/{}".format(args.dataset)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_cond = WrapperCondFlow(model, cond=None)

    for i, (image, mask, masked_image) in enumerate(dataloader):
        masked_image = masked_image.to(device)
        mask = mask.to(device)
        image = image.to(device)
        with torch.no_grad():
            c = first_stage_model.encode(masked_image).latent_dist.sample().mul_(args.scale_factor)
            cc = F.interpolate(mask, size=c.shape[-2:]).to(device, non_blocking=True)

            c = torch.cat((c, cc), dim=1)

            model_cond.cond = c
            z_0 = torch.randn(image.size(0), 4, args.image_size // 8, args.image_size // 8).to(device)
            fake_sample = sample_from_model(model_cond, z_0, args)[-1]
            fake_image = first_stage_model.decode(fake_sample / args.scale_factor).sample

            fake_image = to_range_0_1(fake_image)
            mask = to_range_0_1(mask)
            image = to_range_0_1(image)
            fake_image = fake_image * mask + (1 - mask) * image

            for j, x in enumerate(fake_image):
                index = i * image.size(0) + j
                torchvision.utils.save_image(
                    x, "./inpainting_generated_samples/{}/{}.jpg".format(args.dataset, index), normalized=True
                )
            print("generating batch ", i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ddgan parameters")
    parser.add_argument("--seed", type=int, default=1024, help="seed used for initialization")
    parser.add_argument("--compute_fid", action="store_true", default=False, help="whether or not compute FID")
    parser.add_argument("--epoch_id", type=int, default=500)

    parser.add_argument("--image_size", type=int, default=256, help="size of image")
    parser.add_argument("--num_in_channels", type=int, default=9, help="in channel image")
    parser.add_argument("--num_out_channels", type=int, default=4, help="in channel image")
    parser.add_argument("--nf", type=int, default=256, help="channel of image")

    parser.add_argument("--centered", action="store_false", default=True, help="-1,1 scale")
    parser.add_argument("--resamp_with_conv", type=bool, default=True)
    parser.add_argument("--num_res_blocks", type=int, default=2, help="number of resnet blocks per scale")
    parser.add_argument("--num_heads", type=int, default=4, help="number of head")
    parser.add_argument("--num_head_upsample", type=int, default=-1, help="number of head upsample")
    parser.add_argument("--num_head_channels", type=int, default=-1, help="number of head channels")
    parser.add_argument(
        "--attn_resolutions", nargs="+", type=int, default=(16, 8), help="resolution of applying attention"
    )
    parser.add_argument("--ch_mult", nargs="+", type=int, default=(1, 2, 3, 4), help="channel mult")
    parser.add_argument("--dropout", type=float, default=0.0, help="drop-out rate")
    parser.add_argument("--num_classes", type=int, default=None, help="num classes")
    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)
    parser.add_argument("--scale_factor", type=float, default=0.18215, help="size of image")

    #######################################
    parser.add_argument("--exp", default="latent_kl_exp1", help="name of experiment")
    parser.add_argument(
        "--real_img_dir",
        default="./pytorch_fid/cifar10_train_stat.npy",
        help="directory to real images for FID computation",
    )
    parser.add_argument("--dataset", default="celeba_256", help="name of dataset")
    parser.add_argument("--num_timesteps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=50, help="sample generating batch size")

    # sampling argument
    parser.add_argument("--atol", type=float, default=1e-5, help="absolute tolerance error")
    parser.add_argument("--rtol", type=float, default=1e-5, help="absolute tolerance error")
    parser.add_argument(
        "--method",
        type=str,
        default="dopri5",
        help="solver_method",
        choices=["dopri5", "dopri8", "adaptive_heun", "bosh3", "euler", "midpoint", "rk4"],
    )
    parser.add_argument("--step_size", type=float, default=0.01, help="step_size")
    parser.add_argument("--perturb", action="store_true", default=False)

    parser.add_argument("--pretrained_autoencoder_ckpt", type=str, default="../stabilityai/sd-vae-ft-mse")

    args = parser.parse_args()

    sample_and_test(args)
