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
from sampler import get_sampler

def get_model(config, local_rank, genie=False):
    if genie:
        model_config = config.genie_model
        model = get_genie_model(config).to(config.setup.device)
    else:
        model_config = config.diffusion_model
        model = get_diffusion_model(model_config).to(config.setup.device)

    model = DDP(model, device_ids=[local_rank])
    state = torch.load(model_config.ckpt_path, map_location=config.setup.device)
    logging.info(model.load_state_dict(state['model'], strict=True))
    if 'ema_rate' in model_config.keys():
        ema = EMA(
            model.parameters(), decay=model_config.ema_rate)
        ema.load_state_dict(state['ema'])
        ema.copy_to(model.parameters())
    
    model.eval()
    return model

def sample_batch(sample_dir, counter, max_samples, sampling_fn, sampling_shape, device, labels, n_classes):
    x = torch.randn(sampling_shape, device=device)
    with torch.no_grad():
        if labels is None:
            if n_classes is not None:
                raise ValueError('Need to set labels for class-conditional sampling.')

            x, nfes = sampling_fn(x) 
        else:
            if isinstance(labels, int):
                if labels == n_classes:
                    labels = torch.randint(n_classes, (sampling_shape[0],))
                else:
                    labels = torch.tensor(sampling_shape[0] * [labels], device=x.device)
            elif isinstance(labels, list):
                labels = torch.tensor(labels, device=x.device)

            x, nfes = sampling_fn(x, labels) 

        x = (x / 2. + .5).clip(0., 1.)
        x = x.cpu().permute(0, 2, 3, 1) * 255.
        x = x.numpy().astype(np.uint8)

        if counter == 0:
            logging.info('NFEs: %d' % nfes)

    for img in x:
        if counter < max_samples:
            Image.fromarray(img).save(os.path.join(sample_dir, str(counter).zfill(6) + '.png'))
            counter += 1

    return counter


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
    if 'genie_model' in config.keys() and config.sampler.name == 'ttm2':
        genie_model = get_model(config, local_rank, genie=True)
    else:
        genie_model = None

    if config.diffusion_model.pred == 'eps':
        diffusion_wrapper = EpsPredictor(diffusion_model, config.diffusion_model.M, config.sde.beta_min, config.sde.beta_d)
    elif config.diffusion_model.pred == 'v':
        diffusion_wrapper = VPredictor(diffusion_model, config.diffusion_model.M, config.sde.beta_min, config.sde.beta_d)

    sampling_shape = (config.sampler.batch_size,
                      config.data.num_channels,
                      config.data.image_size,
                      config.data.image_size)
    sampling_fn = get_sampler(config, diffusion_wrapper, genie_model)

    counter = (config.test.n_samples // (sampling_shape[0] * global_size) + 1) * sampling_shape[0] * global_rank
    for _ in range(config.test.n_samples // (sampling_shape[0] * global_size) + 1):
        counter = sample_batch(sample_dir, counter, config.test.n_samples, sampling_fn, sampling_shape, config.setup.device, config.sampler.labels, config.data.n_classes)