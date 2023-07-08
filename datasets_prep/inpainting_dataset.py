import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class InpaintingTrainDataset(Dataset):
    def __init__(self, indir, mask_generator, transform=None):
        self.in_files = list(glob.glob(os.path.join(indir, "**", "*.jpg"), recursive=True))
        self.mask_generator = mask_generator
        self.transform = transform
        self.iter_i = 0

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        img = np.array(Image.open(self.in_files[item]).convert("RGB"))
        img = np.transpose(img, (2, 0, 1))
        mask = self.mask_generator(img, iter_i=self.iter_i)
        mask = torch.from_numpy(mask)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        masked_img = (1.0 - mask) * img
        if self.transform is not None:
            img = self.transform(img)
        self.iter_i += 1
        img = img * 2.0 - 1.0
        mask = mask * 2.0 - 1.0
        masked_img = masked_img * 2.0 - 1.0
        return img, mask, masked_img


def test():
    import torchvision
    from datasets_prep.inpaint_preprocess.mask import get_mask_generator

    mask_gen = get_mask_generator(None, None)
    dataset = InpaintingTrainDataset(indir="dataset/data256x256/", mask_generator=mask_gen)

    img, mask, masked_img = dataset[0]
    mask = mask.reshape(256, 256)
    noise = torch.randn_like(img)
    velocity = img - noise
    # torchvision.utils.save_image(mask, "imask.png", normalize = True)
    torchvision.utils.save_image(img, "image.png", normalize=True)
    torchvision.utils.save_image(velocity, "velocity.png", normalize=True)
