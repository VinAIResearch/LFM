import logging
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from utils.util import make_dir
import sys
import argparse
import importlib
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


def run_main(config):
    processes = []
    for rank in range(config.setup.n_gpus_per_node):
        config.setup.local_rank = rank
        config.setup.global_rank = rank + \
            config.setup.node_rank * config.setup.n_gpus_per_node
        print('Node rank %d, local proc %d, global proc %d' % (
            config.setup.node_rank, config.setup.local_rank, config.setup.global_rank))
        p = mp.Process(target=setup, args=(config, main))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def setup(config, fn):
    os.environ['MASTER_ADDR'] = config.setup.master_address
    os.environ['MASTER_PORT'] = '%d' % config.setup.master_port
    os.environ['OMP_NUM_THREADS'] = '%d' % config.setup.omp_n_threads
    torch.cuda.set_device(config.setup.local_rank)
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            rank=config.setup.global_rank,
                            world_size=config.setup.global_size)
    fn(config)
    dist.barrier()
    dist.destroy_process_group()


def set_logger(gfile_stream):
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')


def main(config):
    workdir = os.path.join(config.setup.root_folder, config.setup.workdir)

    if config.setup.mode == 'train' or config.setup.mode == 'continue':
        if config.setup.global_rank == 0:
            if config.setup.mode == 'train':
                make_dir(workdir)
                gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'w')
            else:
                if not os.path.exists(workdir):
                    raise ValueError('Working directoy does not exist.')
                gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'a')

            set_logger(gfile_stream)
            logging.info(config)

        if config.setup.runner == 'train_diffusion_base':
            from runners import train_diffusion_base
            train_diffusion_base.training(config, workdir, config.setup.mode)
        elif config.setup.runner == 'train_diffusion_upsampler':
            from runners import train_diffusion_upsampler
            train_diffusion_upsampler.training(
                config, workdir, config.setup.mode)
        elif config.setup.runner == 'train_genie_base':
            from runners import train_genie_base
            train_genie_base.training(config, workdir, config.setup.mode)
        elif config.setup.runner == 'train_genie_upsampler':
            from runners import train_genie_upsampler
            train_genie_upsampler.training(config, workdir, config.setup.mode)
        else:
            raise NotImplementedError('Runner is not yet implemented.')

    elif config.setup.mode == 'eval':
        if config.setup.global_rank == 0:
            make_dir(workdir)
            gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'w')
            set_logger(gfile_stream)
            logging.info(config)

        if config.setup.runner == 'generate_base':
            from runners import generate_base
            generate_base.evaluation(config, workdir)
        elif config.setup.runner == 'generate_base_with_guidance':
            from runners import generate_base_with_guidance
            generate_base_with_guidance.evaluation(config, workdir)
        elif config.setup.runner == 'generate_upsampler':
            from runners import generate_upsampler
            generate_upsampler.evaluation(config, workdir)
        else:
            raise NotImplementedError('Runner is not yet implemented.')


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument(
        '--mode', choices=['train', 'continue', 'eval'], required=True)
    parser.add_argument('--workdir', required=True)
    parser.add_argument('--root_folder', default='.')
    parser.add_argument('--n_gpus_per_node', type=int, default=1)
    parser.add_argument('--n_nodes', type=int, default=1)
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--master_address', default='127.0.0.1')
    parser.add_argument('--master_port', type=int, default=6020)
    parser.add_argument('--omp_n_threads', type=int, default=64)

    # Only used for continuing training
    parser.add_argument('--ckpt_path', default=None)
    # Only used for testing
    parser.add_argument('--seed', type=int, default=None)
    # Only used for sampling
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--n_steps', type=int, default=None)
    parser.add_argument('--afs', action='store_true')
    parser.add_argument('--denoising', action='store_true')
    parser.add_argument('--quadratic_striding', action='store_true')
    parser.add_argument('--sampler', choices=['ddim', 'ttm2'], default=None)
    # Only used for conditional sampling while testing
    parser.add_argument('--labels', type=int, default=None)
    # Only used for upsampling
    parser.add_argument('--data_folder', default=None)
    # Only used for guided sampling while testing
    parser.add_argument('--guidance_scale', type=float, default=None)

    args = parser.parse_args()

    config_fn = importlib.import_module('configs.' + args.config)
    config = config_fn.get_config()

    config.setup.mode = args.mode
    config.setup.workdir = args.workdir
    config.setup.root_folder = args.root_folder
    config.setup.n_gpus_per_node = args.n_gpus_per_node
    config.setup.n_nodes = args.n_nodes
    config.setup.node_rank = args.node_rank
    config.setup.master_address = args.master_address
    config.setup.master_port = args.master_port
    config.setup.omp_n_threads = args.omp_n_threads
    config.setup.global_size = config.setup.n_nodes * config.setup.n_gpus_per_node

    if args.ckpt_path is not None:
        if config.setup.mode != 'continue':
            raise ValueError('The ckpt_flag is only used for training continuation.')
        if config.setup.runner == 'train_diffusion_base' or config.setup.runner == 'train_diffusion_upsampler':
            config.diffusion_model.ckpt_path = args.ckpt_path
        elif config.setup.runner == 'train_genie_base' or config.setup.runner == 'train_genie_upsampler':
            config.genie_model.ckpt_path = args.ckpt_path
        else:
            raise NotImplementedError

    if args.seed is not None:
        config.test.seed = args.seed
    if args.batch_size is not None:
        config.sampler.batch_size = args.batch_size
    if args.n_samples is not None:
        if args.n_samples > 100000:
            raise ValueError('Sample in multiple rounds.')
            
        config.test.n_samples = args.n_samples
    if args.n_steps is not None:
        config.sampler.n_steps = args.n_steps
    if args.denoising:
        config.sampler.denoising = True
    if args.afs:
        config.sampler.afs = True
    if args.quadratic_striding:
        config.sampler.quadratic_striding = True
    if args.sampler is not None:
        config.sampler.name = args.sampler
    if args.labels is not None:
        config.sampler.labels = args.labels
    if args.data_folder is not None:
        config.test.data_folder = args.data_folder
    if args.guidance_scale is not None:
        config.sampler.guidance_scale = args.guidance_scale

    run_main(config)
