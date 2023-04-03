import torch
import torch.nn as nn
import functools

from models.score_sde_pytorch.layerspp import ResnetBlockBigGANpp, conv3x3
from utils.util import add_dimensions
from utils.util import get_resize_fn


def get_gamma(t, sde_config):
    alpha_t = (-.5 * (sde_config.beta_min * t + .5 *
               sde_config.beta_d * t ** 2.)).exp()
    return (1. - alpha_t ** 2.).sqrt() / alpha_t


class GENIEModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.all_modules = nn.ModuleList([GENIEPredictionHead(config.genie_model)])

    def forward(self, x, t, eps, xemb, temb, context=None):
        gamma = add_dimensions(get_gamma(t, self.config.sde), len(x.shape) - 1)
        h = self.all_modules[0](x, eps, xemb, temb, context=context)
        head1, head2, head3 = torch.chunk(h, 3, dim=1)
        return -head1 / gamma + head2 * gamma / (1. + gamma ** 2.) + head3 / (gamma * (1. + gamma ** 2.))


class GENIEPredictionHead(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.model_config = model_config
        self.normalize_xemb = model_config.normalize_xemb
        self.apply_act_before_res = model_config.apply_act_before_res
        self.act = act = nn.SiLU()
        self.nf = nf = model_config.nf
        self.num_res_blocks = num_res_blocks = model_config.num_res_blocks
        dropout = model_config.dropout
        fir = model_config.fir
        fir_kernel = model_config.fir_kernel
        self.skip_rescale = skip_rescale = model_config.skip_rescale
        init_scale = model_config.init_scale
        self.image_size = model_config.image_size

        if model_config.is_upsampler:
            self.resize_fn = get_resize_fn(self.image_size)

        ResnetBlock = functools.partial(ResnetBlockBigGANpp,
                                        act=act,
                                        in_ch=nf,
                                        out_ch=nf,
                                        dropout=dropout,
                                        fir=fir,
                                        fir_kernel=fir_kernel,
                                        init_scale=init_scale,
                                        skip_rescale=skip_rescale,
                                        temb_dim=(model_config.num_in_channels - (9 if model_config.is_upsampler else 6)) * 4)

        modules = []
        if self.normalize_xemb:
            modules.append(nn.GroupNorm(num_groups=min(
                (model_config.num_in_channels - 6) // 4, 32), num_channels=model_config.num_in_channels - 6, eps=1e-6))

        modules.append(conv3x3(model_config.num_in_channels,
                       nf, init_scale=init_scale))
        for _ in range(num_res_blocks):
            modules.append(ResnetBlock())

        modules.append(nn.GroupNorm(num_groups=min(
            nf // 4, 32), num_channels=nf, eps=1e-6))
        modules.append(
            conv3x3(nf, model_config.num_out_channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, eps, xemb, temb, context=None):
        if context is not None:
            cond_signal, _ = context
            x = torch.cat((x, self.resize_fn(cond_signal)), dim=1)

        m_idx = 0
        if self.normalize_xemb:
            xemb_normalized = self.all_modules[m_idx](xemb)
            m_idx += 1
            h = torch.cat((x, eps, xemb_normalized), dim=1)
        else:
            h = torch.cat((x, eps, xemb), dim=1)

        if self.apply_act_before_res:
            h = self.act(self.all_modules[m_idx](h))
        else:
            h = self.all_modules[m_idx](h)
        m_idx += 1

        for _ in range(self.num_res_blocks):
            h = self.all_modules[m_idx](h, temb)
            m_idx += 1

        h = self.act(self.all_modules[m_idx](h))
        m_idx += 1
        h = self.all_modules[m_idx](h)
        m_idx += 1

        assert m_idx == len(self.all_modules)
        return h
