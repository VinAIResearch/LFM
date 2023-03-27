import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.setup = ml_collections.ConfigDict()
    config.data = ml_collections.ConfigDict()
    config.sde = ml_collections.ConfigDict()
    config.diffusion_model = ml_collections.ConfigDict()
    config.genie_model = ml_collections.ConfigDict()
    config.sampler = ml_collections.ConfigDict()
    config.test = ml_collections.ConfigDict()

    config.setup.runner = 'generate_base'

    config.data.image_size = 128
    config.data.num_channels = 3
    config.data.n_classes = None

    config.sde.beta_min = .1
    config.sde.beta_d = 19.9

    config.diffusion_model.name = 'ddpm'
    config.diffusion_model.ema_rate = .9999
    config.diffusion_model.num_in_channels = config.data.num_channels
    config.diffusion_model.num_out_channels = config.data.num_channels
    config.diffusion_model.nf = 128
    config.diffusion_model.ch_mult = (1, 1, 2, 2, 4, 4, 4)
    config.diffusion_model.num_res_blocks = 2
    config.diffusion_model.attn_resolutions = (16,)
    config.diffusion_model.resamp_with_conv = True
    config.diffusion_model.dropout = 0.
    config.diffusion_model.image_size = config.data.image_size
    config.diffusion_model.pred = 'eps'
    config.diffusion_model.M = 1.
    config.diffusion_model.ckpt_path = 'work_dir/church/checkpoint_300000.pth'

    config.genie_model.num_in_channels = config.diffusion_model.nf + 2 * config.data.num_channels
    config.genie_model.fir = False
    config.genie_model.fir_kernel = [1, 3, 3, 1]
    config.genie_model.skip_rescale = True
    config.genie_model.init_scale = 0.
    config.genie_model.dropout = 0.
    config.genie_model.num_out_channels = 3 * config.data.num_channels
    config.genie_model.apply_act_before_res = False
    config.genie_model.num_res_blocks = 1
    config.genie_model.normalize_xemb = True
    config.genie_model.nf = config.diffusion_model.nf
    config.genie_model.ckpt_path = 'work_dir/church/genie_checkpoint_35000.pth'
    config.genie_model.is_upsampler = False
    config.genie_model.image_size = config.data.image_size

    config.sampler.name = 'ddim'
    config.sampler.batch_size = 16
    config.sampler.n_steps = 16
    config.sampler.denoising = False
    config.sampler.quadratic_striding = False
    config.sampler.eps = 1e-3
    config.sampler.afs = False
    config.sampler.denoising = False
    config.sampler.labels = None

    config.test.seed = 0
    config.test.n_samples = config.sampler.batch_size

    return config
