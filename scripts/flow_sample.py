import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torchvision
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    model_and_flow_defaults,
    create_model_and_flow,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and flow...")
    model, flow = create_model_and_flow(
        **args_to_dict(args, model_and_flow_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path)
    )
    model.to("cuda")
    model.eval()

    logger.log("sampling...")
    iter_loop = args.num_samples//args.batch_size
    for iter in range(iter_loop):
        x0 = th.randn(args.batch_size, 3, 32, 32).to("cuda")
        sample = flow.decode(model, x0).to("cpu")
        grid = torchvision.utils.make_grid(sample, nrow=4, normalize=True)
        torchvision.utils.save_image(grid, "./sample_{}.png".format(iter))
        logger.log(f"created {iter * args.batch_size} samples")

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        num_samples=64,
        batch_size=16,
        model_path="",
    )
    defaults.update(model_and_flow_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()