from .EDM import get_edm_network
from .DiT import DiT_models
from .guided_diffusion.unet import UNetModel# , UNetModelAttn

def create_network(config):
    if config.use_origin_adm:
        return get_flow_model(config)

    if "DiT" not in config.model_type:
        return get_edm_network(config)
    return DiT_models[config.model_type](
        img_resolution=config.image_size//config.f,
        in_channels=config.num_in_channels,
        label_dropout=config.label_dropout,
        num_classes=config.num_classes
    )


def get_flow_model(config):
    if config.layout:
        model = UNetModelAttn(
                        image_size=config.image_size//8,
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
                        use_spatial_transformer=True,    # custom transformer support
                        transformer_depth=3,              # custom transformer support
                        context_dim=512,                 # custom transformer support
                        legacy=True,
                    )
    else:
        model = UNetModel(image_size=config.image_size//8,
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

    return model
