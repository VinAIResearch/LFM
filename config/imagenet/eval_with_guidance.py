import ml_collections
import copy


def get_config():
    config = ml_collections.ConfigDict()
    config.setup = ml_collections.ConfigDict()
    config.data = ml_collections.ConfigDict()
    config.sde = ml_collections.ConfigDict()
    config.diffusion_model = ml_collections.ConfigDict()
    config.genie_model = ml_collections.ConfigDict()
    config.cond_diffusion_model = ml_collections.ConfigDict()
    config.cond_genie_model = ml_collections.ConfigDict()
    config.sampler = ml_collections.ConfigDict()
    config.test = ml_collections.ConfigDict()

    config.setup.runner = 'generate_base_with_guidance'

    config.data.image_size = 64
    config.data.num_channels = 3
    config.data.n_classes = 1000

    config.sde.beta_min = .1
    config.sde.beta_d = 19.9

    config.diffusion_model.name = 'openai'
    config.diffusion_model.ema_rate = .9999
    config.diffusion_model.num_in_channels = config.data.num_channels
    config.diffusion_model.num_out_channels = config.data.num_channels
    config.diffusion_model.nf = 192
    config.diffusion_model.ch_mult = (1, 2, 3, 4)
    config.diffusion_model.num_res_blocks = 3
    config.diffusion_model.attn_resolutions = (8,)
    config.diffusion_model.resamp_with_conv = True
    config.diffusion_model.dropout = 0.
    config.diffusion_model.image_size = config.data.image_size
    config.diffusion_model.num_heads = None
    config.diffusion_model.num_head_channels = 64
    config.diffusion_model.num_head_upsample = -1
    config.diffusion_model.use_scale_shift_norm = True
    config.diffusion_model.resblock_updown = True
    config.diffusion_model.use_new_attention_order = True
    config.diffusion_model.num_classes = None
    config.diffusion_model.fourier_scale = 16
    config.diffusion_model.pred = 'eps'
    config.diffusion_model.M = 1.
    config.diffusion_model.ckpt_path = 'work_dir/imagenet/checkpoint_400000.pth'

    config.genie_model.num_in_channels = config.diffusion_model.nf + 2 * config.data.num_channels
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
    config.genie_model.ckpt_path = 'work_dir/imagenet/genie_checkpoint_25000.pth'
    config.genie_model.is_upsampler = False
    config.genie_model.image_size = config.data.image_size

    config.cond_diffusion_model = copy.deepcopy(config.diffusion_model)
    config.cond_diffusion_model.ckpt_path = 'work_dir/imagenet/cond_checkpoint_400000.pth'
    config.cond_diffusion_model.num_classes = 1000
    config.cond_genie_model = copy.deepcopy(config.genie_model)
    config.cond_genie_model.ckpt_path = 'work_dir/imagenet/cond_genie_checkpoint_15000.pth'

    config.sampler.name = 'ddim'
    config.sampler.batch_size = 16
    config.sampler.n_steps = 16
    config.sampler.denoising = False
    config.sampler.quadratic_striding = False
    config.sampler.eps = 1e-3
    config.sampler.afs = False
    config.sampler.denoising = False
    config.sampler.labels = None
    config.sampler.guidance_scale = 0.

    config.test.seed = 0
    config.test.n_samples = config.sampler.batch_size

    return config
