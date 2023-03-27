import ml_collections
import copy
import torch.nn as nn
from models.guided_diffusion.unet import UNetModel
from models.guided_diffusion.unet_upsampler import UNetUpsamplerModel
from models.pytorch_diffusion.model import Model
from models.score_sde_pytorch.ncsnpp import NCSNpp
from models.score_sde_pytorch.layerspp import GaussianFourierProjection
from models.score_sde_pytorch.layers import default_init
from models.genie import GENIEModel


def get_diffusion_model(config):
    if config.name == 'openai':
        return get_openai(config)
    elif config.name == 'ddpm':
        return get_ddpm(config)
    elif config.name == 'ncsnpp':
        return get_ncsnpp(config)
    elif config.name == 'openai_upsampler':
        return get_openai_upsampler(config)

def get_genie_model(config):
    return GENIEModel(config)


def get_fourier_embedding(nf, fourier_scale):
    embed_dim = 2 * nf
    time_embed_dim = 4 * nf

    time_embedding = []
    time_embedding.append(GaussianFourierProjection(
        embedding_size=nf, scale=fourier_scale))
    time_embedding.append(nn.Linear(embed_dim, time_embed_dim))
    time_embedding[-1].weight.data = default_init()(time_embedding[-1].weight.shape)
    time_embedding.append(nn.SiLU())
    time_embedding.append(nn.Linear(time_embed_dim, time_embed_dim))
    time_embedding[-1].weight.data = default_init()(time_embedding[-1].weight.shape)

    return nn.Sequential(*time_embedding)


def get_openai(config):
    model = UNetModel(image_size=config.image_size,
                      in_channels=config.num_in_channels,
                      model_channels=config.nf,
                      out_channels=config.num_out_channels,
                      num_res_blocks=config.num_res_blocks,
                      attention_resolutions=config.attn_resolutions,
                      dropout=config.dropout,
                      channel_mult=config.ch_mult,
                      conv_resample=config.resamp_with_conv,
                      dims=2,
                      num_classes=config.num_classes,
                      use_checkpoint=False,
                      use_fp16=False,
                      num_heads=config.num_heads,
                      num_head_channels=config.num_head_channels,
                      num_heads_upsample=config.num_head_upsample,
                      use_scale_shift_norm=config.use_scale_shift_norm,
                      resblock_updown=config.resblock_updown,
                      use_new_attention_order=config.use_new_attention_order)

    if config.fourier_scale is not None:
        model.time_embed = get_fourier_embedding(
            config.nf, config.fourier_scale)

    return model


def get_ddpm(config):
    model = Model(ch=config.nf,
                  out_ch=config.num_out_channels,
                  ch_mult=config.ch_mult,
                  num_res_blocks=config.num_res_blocks,
                  attn_resolutions=config.attn_resolutions,
                  dropout=config.dropout,
                  resamp_with_conv=config.resamp_with_conv,
                  in_channels=config.num_in_channels,
                  resolution=config.image_size)
    return model


def get_ncsnpp(config):
    ncsnpp_config = ml_collections.ConfigDict()
    ncsnpp_config.data = ml_collections.ConfigDict()
    ncsnpp_config.model = copy.deepcopy(config)
    ncsnpp_config.model.nonlinearity = config.act_name
    ncsnpp_config.model.sigma_min = 0.01
    ncsnpp_config.model.sigma_max = 50
    ncsnpp_config.model.num_scales = 1000
    ncsnpp_config.model.scale_by_sigma = False
    ncsnpp_config.data.image_size = config.image_size
    ncsnpp_config.data.num_channels = config.num_in_channels
    ncsnpp_config.data.centered = True
    model = NCSNpp(ncsnpp_config)
    return model

def get_openai_upsampler(config):
    model = UNetUpsamplerModel(image_size=config.image_size,
                               in_channels=config.num_in_channels,
                               model_channels=config.nf,
                               out_channels=config.num_out_channels,
                               num_res_blocks=config.num_res_blocks,
                               attention_resolutions=config.attn_resolutions,
                               dropout=config.dropout,
                               channel_mult=config.ch_mult,
                               conv_resample=config.resamp_with_conv,
                               dims=2,
                               num_classes=config.num_classes,
                               use_checkpoint=False,
                               use_fp16=False,
                               num_heads=config.num_heads,
                               num_head_channels=config.num_head_channels,
                               num_heads_upsample=config.num_head_upsample,
                               use_scale_shift_norm=config.use_scale_shift_norm,
                               resblock_updown=config.resblock_updown,
                               use_new_attention_order=config.use_new_attention_order,
                               fourier_scale=config.fourier_scale)

    return model