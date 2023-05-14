import numpy as np
import lmdb
import os
import io

import torch
import torch.utils.data as data


class LatentDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.train = train
        self.transform = transform
        if self.train:
            latent_path = os.path.join(root, 'train.pkl')
        else:
            latent_path = os.path.join(root, 'val.pkl')
        self.data = torch.load(latent_path)

    def __getitem__(self, index):
        target = self.data["target"][index]
        mean, std = self.data["mean"][index], self.data["std"][index]
        x = mean + torch.randn_like(mean) * std
        
        if self.transform is not None:
            x = self.transform(x)

        return x, target

    def __len__(self):
        return len(self.data["target"])
