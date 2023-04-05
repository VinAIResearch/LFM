import torch
import logging
import torch.distributed as dist
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import make_grid
from torch.autograd.functional import jvp
from models.util import get_flow_model, get_genie_model
from utils.util import set_seeds, make_dir
from utils.optim import get_optimizer
from datasets_prep import get_dataset
from sampler import get_sampler
from dnnlib.util import open_url
from genie_fid.eval import compute_fid


def save_checkpoint(ckpt_path, state):
    saved_state = {'model': state['model'].state_dict(),
                   'optimizer': state['optimizer'].state_dict(),
                   'step': state['step']}
    torch.save(saved_state, ckpt_path)


def _get_flow_model(config):
    model_config = config.flow_model
    model = get_flow_model(model_config).to(config.setup.device)
    state = torch.load(model_config.ckpt_path, map_location=config.setup.device)
    for key in list(state.keys()):
        state[key[7:]] = state.pop(key)
    logging.info(model.load_state_dict(state))
    model.eval()
    return model


def get_state(config, local_rank, mode):
    model_config = config.genie_model
    model = get_genie_model(model_config).to(config.setup.device)
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


def loss_fn(config, flow_model, genie_model, x, y=None):
    t = torch.rand(x.shape[0], device=config.setup.device) * (1.0 - config.train.eps) + config.train.eps
    t = t.view(-1, 1, 1, 1)
    eps = torch.randn_like(x, device=x.device)
    perturbed_data = t * x + (1 - t) * eps
    t = t.squeeze()
    # could we do it in single forward pass, read the jvp function
    _, xemb, temb = flow_model(t, perturbed_data, y=y, return_emb=True)
    deps_dt_backprop = jvp(lambda t: flow_model(t, perturbed_data, y=y), t, v=torch.ones_like(t, device=t.device))[1]
    deps_dt= genie_model(perturbed_data, eps, xemb, temb)
    loss = (deps_dt - deps_dt_backprop) ** 2
    loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
    return loss


def sample_batch(sampling_fn, sampling_shape, device):
    x = torch.randn(sampling_shape, device=device)
    with torch.no_grad():
        x, _ = sampling_fn(x)
        x = (x / 2. + .5).clip(0., 1.)
    return x


def training(config, workdir, mode):
    local_rank = config.setup.local_rank
    global_rank = config.setup.global_rank
    set_seeds(global_rank, config.train.seed)

    torch.cuda.device(local_rank)
    config.setup.device = torch.device('cuda:%d' % local_rank)

    sample_dir = os.path.join(workdir, 'samples/')
    checkpoint_dir = os.path.join(workdir, 'checkpoints/')
    if global_rank == 0 and mode == 'train':
        make_dir(sample_dir)
        make_dir(checkpoint_dir)
    dist.barrier()

    flow_model = _get_flow_model(config)
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
    sampling_fn = get_sampler(config, flow_model, model)
    # loading inception model to compute fid
    with open_url('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl') as f:
        inception_model = pickle.load(f).to(config.setup.device)
        inception_model.eval()
    # loading data
    dataset = get_dataset(config.data)
    dataset_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataset_loader = torch.utils.data.DataLoader(
        dataset=dataset, sampler=dataset_sampler, shuffle=False, batch_size=config.train.batch_size, **config.data.dataloader_params)

    scaler = GradScaler() if config.train.autocast else None
    # setup learning scheduler
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

            x = train_x.to(config.setup.device)
            if config.data.num_classes is None:
                y = None
            else:
                y = train_y.to(config.setup.device)
                if y.dtype == torch.float32:
                    y = y.long()

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=config.train.autocast):
                loss = torch.mean(loss_fn(config, flow_model, model, x, y))

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
