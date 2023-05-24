from models.guided_diffusion.unet import EncoderUNetModel

def classifier_defaults():
    """
    Defaults for classifier models.
    """
    return dict(
        classifier_image_size=32,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",
    )

def create_classifier(
    classifier_image_size,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_fp16,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
):
    if classifier_image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif classifier_image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif classifier_image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif classifier_image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif classifier_image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    out_channel = 1000

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(classifier_image_size // int(res))

    return EncoderUNetModel(
        image_size=classifier_image_size,
        in_channels=4,
        model_channels=classifier_width,
        out_channels=out_channel,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
)
