import logging
import torch


def get_sampler(config, diffusion_wrapper, genie_model):
    if config.sampler.name == 'ddim':
        return get_ddim(config, diffusion_wrapper, genie_model)
    elif config.sampler.name == 'ttm2':
        return get_ttm2(config, diffusion_wrapper, genie_model)
    else:
        raise NotImplementedError


def get_gamma(t, sde_config):
    alpha_t = (-.5 * (sde_config.beta_min * t + .5 *
               sde_config.beta_d * t ** 2.)).exp()
    return (1. - alpha_t ** 2.).sqrt() / alpha_t


def get_ddim(config, diffusion_wrapper, genie_model=None):
    if genie_model is not None:
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

    def ddim(x, y=None, upsampler_signal=None):
        ones = torch.ones(x.shape[0], device=device)
        x_bar = x * (1. + gamma[0] ** 2.).sqrt()
        nfes = 0

        for n in range(n_steps):
            if n == 0 and afs:
                eps = x
            else:
                eps = diffusion_wrapper.eps(x_bar / (1. + gamma[n] ** 2.).sqrt(), ones * t[n], y=y, context=upsampler_signal)
                nfes += 1
            x_bar = x_bar + (gamma[n + 1] - gamma[n]) * eps

        if denoising:
            eps = diffusion_wrapper.eps(x_bar / (1. + gamma[-1] ** 2.).sqrt(), ones * t[-1], y=y, context=upsampler_signal)
            x = x_bar - gamma[-1] * eps
            nfes += 1
        else:
            x = x_bar / (1. + gamma[-1] ** 2.).sqrt()

        return x, nfes

    return ddim


def get_ttm2(config, diffusion_wrapper, genie_model):
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

    def ttm2(x, y=None, upsampler_signal=None):
        ones = torch.ones(x.shape[0], device=device)
        x_bar = x * (1. + gamma[0] ** 2.).sqrt()
        nfes = 0

        for n in range(n_steps):
            if n == 0 and afs:
                eps = x
                deps_dgamma = torch.zeros_like(x, device=x.device)
            else:
                eps, xemb, temb = diffusion_wrapper.eps(x_bar / (1. + gamma[n] ** 2.).sqrt(), ones * t[n], y=y, context=upsampler_signal, return_embeddings=True)
                deps_dgamma = genie_model(x_bar / (1. + gamma[n] ** 2.).sqrt(), ones * t[n], eps, xemb, temb, context=upsampler_signal)
                nfes += 1

            h = (gamma[n + 1] - gamma[n])
            x_bar = x_bar + h * eps + .5 * h ** 2. * deps_dgamma

        if denoising:
            eps = diffusion_wrapper.eps(x_bar / (1. + gamma[-1] ** 2.).sqrt(), ones * t[-1], y=y, context=upsampler_signal)
            x = x_bar - gamma[-1] * eps
            nfes += 1
        else:
            x = x_bar / (1. + gamma[-1] ** 2.).sqrt()

        return x, nfes

    return ttm2