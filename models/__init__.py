from .EDM import get_edm_network
from .DiT import DiT_models

def create_network(config):
    if "DiT" not in config.model_type:
        return get_edm_network(config)

    return DiT_models[config.model_type](
        img_resolution=config.image_size//config.f,
        in_channels=config.num_in_channels,
        label_dropout=config.label_dropout,
        num_classes=config.num_classes
    )