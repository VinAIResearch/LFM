import torch
import logging
import torch.distributed as dist
import numpy as np
import os
import glob
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP

from models.util import get_diffusion_model, get_genie_model
from models.score_sde_pytorch.ema import ExponentialMovingAverage as EMA
from utils.util import set_seeds, make_dir, chunkify, add_dimensions, get_alpha_sigma_fn
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
        logging.info(ema.load_state_dict(state['ema']))
        ema.copy_to(model.parameters())
    
    model.eval()
    return model

def sample_batch(alpha_fn, sigma_fn, global_rank, file_list, low_res_cond, aug_cond, sample_dir, sampling_fn, sampling_shape, device):
    x = torch.randn(sampling_shape, device=device)
    aug_noise = aug_cond * torch.ones(x.shape[0], device=device)
    alpha_t = add_dimensions(alpha_fn(aug_noise), len(x.shape) - 1)
    sigma_t = add_dimensions(sigma_fn(aug_noise), len(x.shape) - 1)
    cond_signal = alpha_t * low_res_cond + sigma_t * torch.randn_like(low_res_cond, device=device)
    upsampler_signal = (cond_signal, aug_noise)
    with torch.no_grad():
        x, nfes = sampling_fn(x, upsampler_signal=upsampler_signal) 
        x = (x / 2. + .5).clip(0., 1.)
        x = x.cpu().permute(0, 2, 3, 1) * 255.
        x = x.numpy().astype(np.uint8)

        if global_rank == 0 and round == 0:
            logging.info('NFEs: %d' % nfes)

    for i, img in enumerate(x):
        Image.fromarray(img).save(os.path.join(sample_dir, file_list[i].split('/')[-1]))

    return


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

    alpha_fn, sigma_fn = get_alpha_sigma_fn(config.sde.beta_min, config.sde.beta_d)

    sampling_shape = (config.sampler.batch_size,
                      config.data.num_channels,
                      config.data.image_size,
                      config.data.image_size)
    sampling_fn = get_sampler(config, diffusion_wrapper, genie_model)

    data_folder = config.test.data_folder
    all_data_files = []
    for file in sorted(glob.glob(data_folder + '/*')):
        if file.endswith('.png'):
            all_data_files.append(file)
        else:
            raise ValueError('Folder contains non-png files.')

    if len(all_data_files) > 100000:
        raise ValueError('Sample in multiple rounds.')

    local_file_list = chunkify(all_data_files, global_size)[global_rank]
    n_rounds = (len(local_file_list) // sampling_shape[0]) + 1
    for i in range(n_rounds):
        file_list = local_file_list[i*sampling_shape[0]:(i+1)*sampling_shape[0]]
        files = []
        for file in file_list:
            img = Image.open(file)
            files.append(torch.tensor(np.array(img).astype(np.float32) / 127.5 - 1., device=config.setup.device).permute(2, 0, 1))

        if len(files) > 0:
            low_res_cond = torch.stack(files)
            sample_batch(alpha_fn, sigma_fn, global_rank, file_list, low_res_cond, config.sampler.aug_noise, sample_dir, sampling_fn, sampling_shape, config.setup.device)