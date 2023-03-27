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

    config.setup.runner = 'generate_upsampler'

    config.data.image_size = 512
    config.data.num_channels = 3
    config.data.n_classes = None

    config.sde.beta_min = .1
    config.sde.beta_d = 19.9

    config.diffusion_model.name = 'openai_upsampler'
    config.diffusion_model.ema_rate = 0.9999
    config.diffusion_model.num_in_channels = 2 * config.data.num_channels
    config.diffusion_model.num_out_channels = config.data.num_channels
    config.diffusion_model.nf = 96
    config.diffusion_model.ch_mult = (1, 1, 2, 2, 3, 3, 4)
    config.diffusion_model.num_res_blocks = 2
    config.diffusion_model.attn_resolutions = (32, 64)
    config.diffusion_model.resamp_with_conv = True
    config.diffusion_model.dropout = .1
    config.diffusion_model.image_size = config.data.image_size
    config.diffusion_model.num_heads = None
    config.diffusion_model.num_head_channels = 32
    config.diffusion_model.num_head_upsample = -1
    config.diffusion_model.use_scale_shift_norm = True
    config.diffusion_model.resblock_updown = True
    config.diffusion_model.use_new_attention_order = True
    config.diffusion_model.num_classes = None
    config.diffusion_model.fourier_scale = 16
    config.diffusion_model.pred = 'v'
    config.diffusion_model.M = 1.
    config.diffusion_model.cond_res = config.data.image_size // 4
    config.diffusion_model.ckpt_path = 'work_dir/cats/upsampler/checkpoint_150000.pth'

    config.genie_model.num_in_channels = config.diffusion_model.nf + 3 * config.data.num_channels
    config.genie_model.fir = False
    config.genie_model.fir_kernel = [1, 3, 3, 1]
    config.genie_model.skip_rescale = True
    config.genie_model.init_scale = 0.
    config.genie_model.dropout = 0.
    config.genie_model.num_out_channels = 3 * config.data.num_channels
    config.genie_model.apply_act_before_res = False
    config.genie_model.num_res_blocks = 1
    config.genie_model.normalize_xemb = False
    config.genie_model.nf = config.diffusion_model.nf
    config.genie_model.ckpt_path = 'work_dir/cats/upsampler/genie_checkpoint_20000.pth'
    config.genie_model.is_upsampler = True
    config.genie_model.image_size = config.data.image_size

    config.sampler.name = 'ddim'
    config.sampler.batch_size = 1
    config.sampler.n_steps = 16
    config.sampler.denoising = False
    config.sampler.quadratic_striding = False
    config.sampler.eps = 1e-3
    config.sampler.afs = False
    config.sampler.denoising = False
    config.sampler.aug_noise = .1
    config.sampler.labels = None

    config.test.seed = 0
    config.test.data_folder = 'none'

    return config
