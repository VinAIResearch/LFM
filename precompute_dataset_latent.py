import argparse
import os

import numpy as np
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
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

    os.makedirs(args.save_path, exist_ok=True)
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

    for i, (x, y) in enumerate(tqdm(dataloader)):
        if i < 2805:
            continue
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            z = first_stage_model.encode(x).latent_dist.sample().mul_(0.18215)

        z = z.detach().cpu().numpy() # (1, 4, 32, 32)
        y = y.numpy() # (1,)
        for j in range(len(z)):
            np.save(f'{args.save_path}/{str(i * args.batch_size + j).zfill(9)}.npy', {"input": z[j], "label": y[j]})
        print('Generate batch {}'.format(i))


    # test
    debug_idex = list(range(0, 50000, 12500))
    data = [np.load(f"{args.save_path}/{str(i).zfill(9)}.npy", allow_pickle=True) for i in debug_idex]
    sample = torch.cat([torch.from_numpy(x.item()["input"]) for x in data])
    with torch.no_grad():
        rec_image = first_stage_model.decode(sample.cuda()).sample
    rec_image = torch.clamp((rec_image + 1.) / 2., 0, 1)
    torchvision.utils.save_image(rec_image, './rec_debug.jpg')
