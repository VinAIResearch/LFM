import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.setup = ml_collections.ConfigDict()
    config.data = ml_collections.ConfigDict()
    config.data.dataset_params = ml_collections.ConfigDict()
    config.data.dataloader_params = ml_collections.ConfigDict()
    config.sde = ml_collections.ConfigDict()
    config.diffusion_model = ml_collections.ConfigDict()
    config.genie_model = ml_collections.ConfigDict()
    config.sampler = ml_collections.ConfigDict()
    config.optim = ml_collections.ConfigDict()
    config.optim.params = ml_collections.ConfigDict()
    config.train = ml_collections.ConfigDict()

    config.setup.runner = 'train_genie_base'

    config.data.image_size = 128
    config.data.num_channels = 3
    config.data.fid_stats = ['assets/stats/bedroom_50k.npz']
    config.data.path = 'data/processed/bedroom.zip'
    config.data.num_classes = None
    config.data.dataset_params.xflip = True
    config.data.dataloader_params.num_workers = 1
    config.data.dataloader_params.pin_memory = True
    config.data.dataloader_params.drop_last = True

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
    config.diffusion_model.ckpt_path = 'work_dir/bedroom/checkpoint_300000.pth'

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
    config.genie_model.is_upsampler = False
    config.genie_model.image_size = config.data.image_size
    config.genie_model.ckpt_path = None

    config.sampler.name = 'ttm2'
    config.sampler.batch_size = 16
    config.sampler.n_steps = 16
    config.sampler.denoising = False
    config.sampler.quadratic_striding = False
    config.sampler.eps = 1e-3
    config.sampler.afs = False
    config.sampler.denoising = False

    config.optim.optimizer = 'Adam'
    config.optim.params.learning_rate = 1e-4
    config.optim.params.weight_decay = 0.
    config.optim.params.grad_clip = 1.
    config.optim.decay_scheduler = None

    config.train.seed = 0
    config.train.eps = 1e-3
    config.train.n_iters = 40000
    config.train.n_warmup_iters = 10000
    config.train.batch_size = 32
    config.train.autocast = False
    config.train.log_freq = 100
    config.train.snapshot_freq = 1000
    config.train.snapshot_threshold = 1
    config.train.save_freq = 5000
    config.train.save_threshold = 1
    config.train.fid_freq = 5000
    config.train.fid_threshold = 1
    config.train.fid_samples = 10000

    return config