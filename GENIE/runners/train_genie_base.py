import torch
import logging
import torch.distributed as dist
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import make_grid
from torch.autograd.functional import jvp

from models.util import get_diffusion_model, get_genie_model
from models.score_sde_pytorch.ema import ExponentialMovingAverage as EMA
from utils.util import set_seeds, make_dir, get_alpha_sigma_fn, add_dimensions
from utils.optim import get_optimizer
from utils.image_dataset import ImageFolderDataset
from wrappers import EpsPredictor, VPredictor
from sampler import get_sampler
from dnnlib.util import open_url
from eval import compute_fid


def save_checkpoint(ckpt_path, state):
    saved_state = {'model': state['model'].state_dict(),
                   'optimizer': state['optimizer'].state_dict(),
                   'step': state['step']}
    torch.save(saved_state, ckpt_path)

def _get_diffusion_model(config, local_rank):
    model_config = config.diffusion_model
    model = get_diffusion_model(model_config).to(config.setup.device)

    model = DDP(model, device_ids=[local_rank])
    state = torch.load(model_config.ckpt_path, map_location=config.setup.device)
    logging.info(model.load_state_dict(state['model'], strict=True))
    ema = EMA(
        model.parameters(), decay=model_config.ema_rate)
    ema.load_state_dict(state['ema'])
    ema.copy_to(model.parameters())
    model.eval()
    return model.module


def get_state(config, local_rank, mode):
    model_config = config.genie_model
    model = get_genie_model(config).to(config.setup.device)

    model = DDP(model, device_ids=[local_rank])
    optimizer = get_optimizer(config.optim.optimizer, model.parameters(), **config.optim.params)
    step = 0

    if mode == 'continue':
        loaded_state = torch.load(
            model_config.ckpt_path, map_location=config.setup.device)
        model.load_state_dict(loaded_state['model'], strict=False)
        optimizer.load_state_dict(loaded_state['optimizer'])
        step = loaded_state['step']

    return model, optimizer, step


def get_loss_fn(config):
    alpha_fn, sigma_fn = get_alpha_sigma_fn(
        config.sde.beta_min, config.sde.beta_d)

    def loss_fn(diffusion_wrapper, genie_model, x, y=None):
        t = torch.rand(x.shape[0], device=config.setup.device) * \
            (1.0 - config.train.eps) + config.train.eps
        eps = torch.randn_like(x, device=x.device)

        alpha_t = add_dimensions(alpha_fn(t), len(x.shape) - 1)
        sigma_t = add_dimensions(sigma_fn(t), len(x.shape) - 1)
        gamma_t = alpha_t / sigma_t
        a = 2. * gamma_t / (config.sde.beta_d * (gamma_t ** 2. + 1))
        b = torch.sqrt((config.sde.beta_min / config.sde.beta_d) ** 2. + 2. * torch.log(gamma_t ** 2. + 1.) / config.sde.beta_d)
        dt_dgamma_t =  a / b

        perturbed_data = alpha_t * x + sigma_t * eps
        eps_pred, xemb, temb = diffusion_wrapper.eps(perturbed_data, t, y=y, return_embeddings=True)

        eval_point = eps_pred / torch.sqrt(1. + gamma_t ** 2.) - perturbed_data * gamma_t / (1. + gamma_t ** 2.)
        deps_dx_backprop = jvp(lambda x: diffusion_wrapper.eps(x, t, y=y), perturbed_data, eval_point)[1]
        deps_dt_backprop = jvp(lambda t: diffusion_wrapper.eps(perturbed_data, t, y=y), t, v=torch.ones_like(t, device=t.device))[1] * dt_dgamma_t
        deps_dgamma_backprop = deps_dt_backprop + deps_dx_backprop

        deps_dgamma = genie_model(perturbed_data, t, eps, xemb, temb)

        loss = (deps_dgamma - deps_dgamma_backprop) ** 2. * gamma_t ** 2.
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
        return loss
    return loss_fn


def sample_batch(sampling_fn, sampling_shape, device):
    x = torch.randn(sampling_shape, device=device)
    with torch.no_grad():
        x, _ = sampling_fn(x)
        x = (x / 2. + .5).clip(0., 1.)
    return x


def training(config, workdir, mode):
    local_rank = config.setup.local_rank
    global_rank = config.setup.global_rank
    global_size = config.setup.global_size
    set_seeds(global_rank, config.train.seed)

    torch.cuda.device(local_rank)
    config.setup.device = torch.device('cuda:%d' % local_rank)

    sample_dir = os.path.join(workdir, 'samples/')
    checkpoint_dir = os.path.join(workdir, 'checkpoints/')
    if global_rank == 0 and mode == 'train':
        make_dir(sample_dir)
        make_dir(checkpoint_dir)
    dist.barrier()

    diffusion_model = _get_diffusion_model(config, local_rank)
    if config.diffusion_model.pred == 'eps':
        diffusion_wrapper = EpsPredictor(
            diffusion_model, config.diffusion_model.M, config.sde.beta_min, config.sde.beta_d)
    elif config.diffusion_model.pred == 'v':
        diffusion_wrapper = VPredictor(
            diffusion_model, config.diffusion_model.M, config.sde.beta_min, config.sde.beta_d)

    model, optimizer, step = get_state(config, local_rank, mode)
    state = dict(model=model, optimizer=optimizer, step=step)

    if mode == 'continue':
        config.train.snapshot_threshold = step + 1
        config.train.save_threshold = step + 1
        config.train.fid_threshold = step + 1

    sampling_shape = (config.sampler.batch_size,
                      config.data.num_channels,
                      config.data.image_size,
                      config.data.image_size)
    sampling_fn = get_sampler(config, diffusion_wrapper, model)

    with open_url('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl') as f:
        inception_model = pickle.load(f).to(config.setup.device)
        inception_model.eval()

    dataset = ImageFolderDataset(
        config.data.path, config.data.image_size, **config.data.dataset_params)
    dataset_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataset_loader = torch.utils.data.DataLoader(
        dataset=dataset, sampler=dataset_sampler, shuffle=False, batch_size=config.train.batch_size, **config.data.dataloader_params)

    loss_fn = get_loss_fn(config)

    scaler = GradScaler() if config.train.autocast else None

    if config.optim.decay_scheduler is not None:
        if config.train.n_warmup_iters > 0:
            raise ValueError('For now, let us not combine warmup and decay.')
        lambda_fn = lambda step: 1. - float(step) / config.optim.decay_scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_fn)

    while state['step'] < config.train.n_iters:
        dataset_loader.sampler.set_epoch(state['step'] + config.train.seed)

        for _, (train_x, train_y) in enumerate(dataset_loader):
            if state['step'] >= config.train.n_iters:
                break

            if state['step'] % config.train.snapshot_freq == 0 and state['step'] >= config.train.snapshot_threshold and config.setup.global_rank == 0:
                logging.info(
                    'Saving snapshot checkpoint and sampling single batch at iteration %d.' % state['step'])

                model.eval()
                with torch.no_grad():
                    x = sample_batch(sampling_fn, sampling_shape, device=config.setup.device)

                    nrow = int(np.sqrt(x.shape[0]))
                    image_grid = make_grid(x, nrow)
                    plt.figure(figsize=(6, 6))
                    plt.axis('off')
                    plt.imshow(image_grid.permute(1, 2, 0).cpu())
                    plt.savefig(os.path.join(sample_dir, 'iter_%d.png' % state['step']))
                    plt.close()
                model.train()

                save_checkpoint(os.path.join(
                    checkpoint_dir, 'snapshot_checkpoint.pth'), state)
            dist.barrier()

            if state['step'] % config.train.fid_freq == 0 and state['step'] >= config.train.fid_threshold:
                model.eval()
                with torch.no_grad():
                    fid_list = compute_fid(config.train.fid_samples, config.setup.global_size, sampling_shape, sampling_fn, inception_model,
                                             config.data.fid_stats, config.setup.device, config.data.num_classes)

                    if config.setup.global_rank == 0:
                        for i, fid in enumerate(fid_list):
                            logging.info('FID (%d) at step %d: %.6f' % (i + 1, state['step'], fid))
                    dist.barrier()
                model.train()

            if state['step'] % config.train.save_freq == 0 and state['step'] >= config.train.save_threshold and config.setup.global_rank == 0:
                checkpoint_file = os.path.join(
                    checkpoint_dir, 'checkpoint_%d.pth' % state['step'])
                save_checkpoint(checkpoint_file, state)
                logging.info(
                    'Saving  checkpoint at iteration %d' % state['step'])
            dist.barrier()

            x = (train_x.to(config.setup.device).to(torch.float32) / 127.5 - 1.)
            if config.data.num_classes is None:
                y = None
            else:
                y = train_y.to(config.setup.device)
                if y.dtype == torch.float32:
                    y = y.long()

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=config.train.autocast):
                loss = torch.mean(loss_fn(diffusion_wrapper, model, x, y))

            if config.train.n_warmup_iters > 0 and state['step'] <= config.train.n_warmup_iters:
                for g in optimizer.param_groups:
                    g['lr'] = config.optim.params.learning_rate * \
                        np.minimum(state['step'] /
                                   config.train.n_warmup_iters, 1.0)

            if config.train.autocast:
                scaler.scale(loss).backward()
                if config.optim.params.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=config.optim.params.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config.optim.params.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=config.optim.params.grad_clip)
                optimizer.step()

            if (state['step'] + 1) % config.train.log_freq == 0 and config.setup.global_rank == 0:
                logging.info('Loss: %.4f, step: %d' %
                             (loss, state['step'] + 1))
            dist.barrier()

            if config.optim.decay_scheduler is not None:
                scheduler.step()

            state['step'] += 1
