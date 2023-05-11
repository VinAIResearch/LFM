import argparse
import os

import numpy as np
import torch
import torchvision
from datasets_prep import get_dataset
from pytorch_fid.fid_score import compute_statistics_of_path
from pytorch_fid.inception import InceptionV3
from tqdm import tqdm

from diffusers.models import AutoencoderKL

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Compute dataset stat')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--datadir', default='./data')
    parser.add_argument(
        '--save_path', default='./pytorch_fid/cifar10_stat.npy')

    parser.add_argument('--image_size', type=int, default=32,
                        help='size of image')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='size of image')

    parser.add_argument('--pretrained_autoencoder_ckpt', type=str, default="stabilityai/sd-vae-ft-mse")

    args = parser.parse_args()

    device = 'cuda:0'
    dataset = get_dataset(args)

    save_dir = "./real_samples/{}/".format(args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=4,  # cpu_count(),
                                             )
    first_stage_model = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device)
    first_stage_model = first_stage_model.eval()
    first_stage_model.train = False
    for param in first_stage_model.parameters():
        param.requires_grad = False

    # outputs = {"mean": [], "std": [], "target": []}
    # for i, (x, y) in enumerate(tqdm(dataloader)):
    #     x = x.to(device, non_blocking=True)
    #     z = first_stage_model.encode(x).latent_dist # .sample().mul_(args.scale_factor)
    #     outputs["mean"].append(z.mean.detach().cpu())
    #     outputs["std"].append(z.std.detach().cpu())
    #     outputs["target"].append(y)
    #     print('Generate batch {}'.format(i))

    # outputs["mean"] = torch.cat(outputs["mean"])
    # outputs["std"] = torch.cat(outputs["std"])
    # outputs["target"] = torch.cat(outputs["target"])

    save_path = args.save_path
    # torch.save(outputs, save_path)
    # print("Save latents in {}".format(save_path))

    # test
    data = torch.load(save_path)
    print(f"Mean: {data['mean'].shape}, std: {data['std'].shape}")
    print(f"Target: {data['target']}")

    sample = data['mean'][:4] + data['std'][:4] * torch.randn_like(data['mean'][:4])
    rec_image = first_stage_model.decode(sample.cuda()).sample
    rec_image = torch.clamp((rec_image + 1.) / 2., 0, 1)
    torchvision.utils.save_image(rec_image, './rec_debug.jpg')
    
