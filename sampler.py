import logging
import torch


def get_sampler(config, flow_model, genie_model):
    if config.sampler.name == 'ttm2':
        return get_ttm2(config, flow_model, genie_model)
    else:
        raise NotImplementedError


def get_ttm2(config, flow_model, genie_model):
    device = config.setup.device
    n_steps = config.sampler.n_steps
    t_final = config.sampler.eps
    t_start = 1.

    if config.sampler.quadratic_striding:
        t = torch.linspace(t_start ** .5, t_final ** .5,
                           n_steps + 1, device=device) ** 2.
    else:
        t = torch.linspace(t_start, t_final, n_steps + 1, device=device)

    def ttm2(x, y=None):
        ones = torch.ones(x.shape[0], device=device)
        nfes = 0
        for n in range(n_steps):
            eps, xemb, temb = flow_model(ones * t[n], x, y=y, return_emb=True)
            deps_dt = genie_model(x, eps, xemb, temb)
            nfes += 1
            h = (t[n + 1] - t[n])
            x = x + h * eps + .5 * h ** 2. * deps_dt
        return x, nfes
    return ttm2