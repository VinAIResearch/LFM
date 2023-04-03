import torch
import logging
import torch.distributed as dist
import numpy as np
import os
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP

from models.util import get_diffusion_model, get_genie_model
from models.score_sde_pytorch.ema import ExponentialMovingAverage as EMA
from utils.util import set_seeds, make_dir
from wrappers import EpsPredictor, VPredictor
from sampler import get_gamma


def get_model(config, local_rank, genie=False, conditional=False):
    if genie:
        model_config = config.genie_model if not conditional else config.cond_genie_model
        model = get_genie_model(config).to(config.setup.device)
    else:
        model_config = config.diffusion_model if not conditional else config.cond_diffusion_model
        model = get_diffusion_model(model_config).to(config.setup.device)

    model = DDP(model, device_ids=[local_rank])
    state = torch.load(model_config.ckpt_path,
                       map_location=config.setup.device)
    logging.info(model.load_state_dict(state['model'], strict=True))
    if 'ema_rate' in model_config.keys():
        ema = EMA(
            model.parameters(), decay=model_config.ema_rate)
        ema.load_state_dict(state['ema'])
        ema.copy_to(model.parameters())

    model.eval()
    return model


def sample_batch(sample_dir, counter, max_samples, sampling_fn, sampling_shape, device, labels, n_classes, guidance_scale):
    x = torch.randn(sampling_shape, device=device)
    with torch.no_grad():
        if labels is None:
            raise ValueError('Need to set labels for guided class-conditional sampling.')
        else:
            if isinstance(labels, int):
                if labels == n_classes:
                    labels = torch.randint(n_classes, (sampling_shape[0],))
                else:
                    labels = torch.tensor(sampling_shape[0] * [labels], device=x.device)
            elif isinstance(labels, list):
                labels = torch.tensor(labels, device=x.device)

            if guidance_scale == 0.:
                raise ValueError('Set guidance scale > 0.')

            x, nfes = sampling_fn(x, guidance_scale, labels) 

        x = (x / 2. + .5).clip(0., 1.)
        x = x.cpu().permute(0, 2, 3, 1) * 255.
        x = x.numpy().astype(np.uint8)

        if counter == 0:
            logging.info('NFEs: %d' % nfes)

    for img in x:
        if counter < max_samples:
            Image.fromarray(img).save(os.path.join(
                sample_dir, str(counter).zfill(6) + '.png'))
            counter += 1

    return counter


def get_sampler(config, diffusion_wrapper, cond_diffusion_wrapper, genie_model, cond_genie_model):
    if config.sampler.name == 'ddim':
        return get_ddim(config, diffusion_wrapper, cond_diffusion_wrapper, genie_model, cond_genie_model)
    elif config.sampler.name == 'ttm2':
        return get_ttm2(config, diffusion_wrapper, cond_diffusion_wrapper, genie_model, cond_genie_model)
    else:
        raise NotImplementedError


def get_ddim(config, diffusion_wrapper, cond_diffusion_wrapper, genie_model=None, cond_genie_model=None):
    if genie_model is not None or cond_genie_model is not None:
        logging.info(
            'You provided a genie model, but this solver does not use it.')

    device = config.setup.device
    n_steps = config.sampler.n_steps
    t_final = config.sampler.eps
    afs = config.sampler.afs
    denoising = config.sampler.denoising
    t_start = 1.

    if config.sampler.quadratic_striding:
        t = torch.linspace(t_start ** .5, t_final ** .5,
                           n_steps + 1, device=device) ** 2.
    else:
        t = torch.linspace(t_start, t_final, n_steps + 1, device=device)
    gamma = get_gamma(t, config.sde)

    def ddim(x, guidance_scale, y=None, upsampler_signal=None):
        if y is None:
            raise ValueError('Need to set label for guided DDIM.')

        ones = torch.ones(x.shape[0], device=device)
        x_bar = x * (1. + gamma[0] ** 2.).sqrt()
        nfes = 0

        for n in range(n_steps):
            if n == 0 and afs:
                eps = x
            else:
                eps = diffusion_wrapper.eps(
                    x_bar / (1. + gamma[n] ** 2.).sqrt(), ones * t[n], context=upsampler_signal)
                cond_eps = cond_diffusion_wrapper.eps(
                    x_bar / (1. + gamma[n] ** 2.).sqrt(), ones * t[n], y=y, context=upsampler_signal)
                nfes += 1
            x_bar = x_bar + (gamma[n + 1] - gamma[n]) * \
                ((1. + guidance_scale) * cond_eps - guidance_scale * eps)

        if denoising:
            eps = diffusion_wrapper.eps(
                x_bar / (1. + gamma[-1] ** 2.).sqrt(), ones * t[-1], context=upsampler_signal)
            cond_eps = cond_diffusion_wrapper.eps(
                x_bar / (1. + gamma[n] ** 2.).sqrt(), ones * t[n], y=y, context=upsampler_signal)
            x = x_bar - gamma[-1] * ((1. + guidance_scale)
                                     * cond_eps - guidance_scale * eps)
            nfes += 1
        else:
            x = x_bar / (1. + gamma[-1] ** 2.).sqrt()

        return x, nfes

    return ddim


def get_ttm2(config, diffusion_wrapper, cond_diffusion_wrapper, genie_model=None, cond_genie_model=None):
    device = config.setup.device
    n_steps = config.sampler.n_steps
    t_final = config.sampler.eps
    afs = config.sampler.afs
    denoising = config.sampler.denoising
    t_start = 1.

    if config.sampler.quadratic_striding:
        t = torch.linspace(t_start ** .5, t_final ** .5,
                           n_steps + 1, device=device) ** 2.
    else:
        t = torch.linspace(t_start, t_final, n_steps + 1, device=device)
    gamma = get_gamma(t, config.sde)

    def ttm2(x, guidance_scale, y=None, upsampler_signal=None):
        if y is None:
            raise ValueError('Need to set label for guided DDIM.')

        ones = torch.ones(x.shape[0], device=device)
        x_bar = x * (1. + gamma[0] ** 2.).sqrt()
        nfes = 0

        for n in range(n_steps):
            if n == 0 and afs:
                eps = x
                deps_dgamma = torch.zeros_like(x, device=x.device)
            else:
                eps, xemb, temb = diffusion_wrapper.eps(x_bar / (1. + gamma[n] ** 2.).sqrt(
                ), ones * t[n], context=upsampler_signal, return_embeddings=True)
                cond_eps, cond_xemb, cond_temb = cond_diffusion_wrapper.eps(
                    x_bar / (1. + gamma[n] ** 2.).sqrt(), ones * t[n], y=y, context=upsampler_signal, return_embeddings=True)
                deps_dgamma = genie_model(
                    x_bar / (1. + gamma[n] ** 2.).sqrt(), ones * t[n], eps, xemb, temb, context=upsampler_signal)
                cond_deps_dgamma = genie_model(x_bar / (1. + gamma[n] ** 2.).sqrt(
                ), ones * t[n], cond_eps, cond_xemb, cond_temb, context=upsampler_signal)
                nfes += 1

            h = (gamma[n + 1] - gamma[n])
            x_bar = x_bar + h * ((1. + guidance_scale) * cond_eps - guidance_scale * eps) + .5 * \
                h ** 2. * ((1. + guidance_scale) *
                           cond_deps_dgamma - guidance_scale * deps_dgamma)

        if denoising:
            eps = diffusion_wrapper.eps(
                x_bar / (1. + gamma[-1] ** 2.).sqrt(), ones * t[-1], context=upsampler_signal)
            cond_eps = cond_diffusion_wrapper.eps(
                x_bar / (1. + gamma[n] ** 2.).sqrt(), ones * t[n], y=y, context=upsampler_signal)
            x = x_bar - gamma[-1] * ((1. + guidance_scale)
                                     * cond_eps - guidance_scale * eps)
            nfes += 1
        else:
            x = x_bar / (1. + gamma[-1] ** 2.).sqrt()

        return x, nfes

    return ttm2


def evaluation(config, workdir):
    local_rank = config.setup.local_rank
    global_rank = config.setup.global_rank
    global_size = config.setup.global_size
    set_seeds(global_rank, config.test.seed)

    torch.cuda.device(local_rank)
    config.setup.device = torch.device('cuda:%d' % local_rank)

    sample_dir = os.path.join(workdir, 'samples/')
    if global_rank == 0:
        make_dir(sample_dir)
    dist.barrier()

    diffusion_model = get_model(config, local_rank)
    cond_diffusion_model = get_model(config, local_rank, conditional=True)
    if 'genie_model' in config.keys() and config.sampler.name == 'ttm2':
        genie_model = get_model(config, local_rank, genie=True)
        cond_genie_model = get_model(
            config, local_rank, genie=True, conditional=True)
    else:
        genie_model = None
        cond_genie_model = None

    if config.diffusion_model.pred == 'eps':
        diffusion_wrapper = EpsPredictor(
            diffusion_model, config.diffusion_model.M, config.sde.beta_min, config.sde.beta_d)
        cond_diffusion_wrapper = EpsPredictor(
            cond_diffusion_model, config.cond_diffusion_model.M, config.sde.beta_min, config.sde.beta_d)
    elif config.diffusion_model.pred == 'v':
        diffusion_wrapper = VPredictor(
            diffusion_model, config.diffusion_model.M, config.sde.beta_min, config.sde.beta_d)
        cond_diffusion_wrapper = VPredictor(
            cond_diffusion_model, config.cond_diffusion_model.M, config.sde.beta_min, config.sde.beta_d)

    sampling_shape = (config.sampler.batch_size,
                      config.data.num_channels,
                      config.data.image_size,
                      config.data.image_size)
    sampling_fn = get_sampler(
        config, diffusion_wrapper, cond_diffusion_wrapper, genie_model, cond_genie_model)

    counter = (config.test.n_samples //
               (sampling_shape[0] * global_size) + 1) * sampling_shape[0] * global_rank
    for _ in range(config.test.n_samples // (sampling_shape[0] * global_size) + 1):
        counter = sample_batch(sample_dir, counter, config.test.n_samples,
                               sampling_fn, sampling_shape, config.setup.device, config.sampler.labels, config.data.n_classes, config.sampler.guidance_scale)
