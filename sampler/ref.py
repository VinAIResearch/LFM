def get_euler(config, flow_model, genie_model):
    device = config.setup.device
    n_steps = config.sampler.n_steps
    t_final = config.sampler.eps
    t_start = 1.

    if config.sampler.quadratic_striding:
        t = torch.linspace(t_start ** .5, t_final ** .5,
                           n_steps + 1, device=device) ** 2.
    else:
        t = torch.linspace(t_start, t_final, n_steps + 1, device=device)

    def euler(x, y=None):
        ones = torch.ones(x.shape[0], device=device)
        nfes = 0
        for n in range(n_steps):
            eps = flow_model(ones * t[n], x, y=y)
            nfes += 1
            h = (t[n + 1] - t[n])
            x = x + h * eps
        return x, nfes
    return euler 