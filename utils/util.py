import os
import torch
import numpy as np
import torchvision
import torch.distributed as dist

def average_tensor(t):
    size = float(dist.get_world_size())
    dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
    t.data /= size

def get_alpha_sigma_fn(beta_min, beta_d):
    def get_alpha_t(t):
        return (-.5 * (beta_min * t + .5 * beta_d * t ** 2.)).exp()

    def get_sigma_t(t):
        return (1. - get_alpha_t(t) ** 2.).sqrt()

    return get_alpha_t, get_sigma_t


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


def get_resize_fn(res):
    def resize_fn(x):
        return torchvision.transforms.functional.resize(x, res, antialias=True)

    return resize_fn


def add_dimensions(x, n_additional_dims):
    for _ in range(n_additional_dims):
        x = x.unsqueeze(-1)

    return x


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        raise ValueError('Directory already exists.')


def set_seeds(rank, seed):
    torch.manual_seed(rank + seed)
    np.random.seed(rank + seed)
    torch.cuda.manual_seed(rank + seed)
    torch.cuda.manual_seed_all(rank + seed)
    torch.backends.cudnn.benchmark = True
