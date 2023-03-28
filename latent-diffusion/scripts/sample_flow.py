import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange
from torchdiffeq import odeint_adjoint as odeint
from omegaconf import OmegaConf
from PIL import Image
from pytorch_fid.fid_score import calculate_fid_given_paths
from ldm.util import instantiate_from_config

rescale = lambda x: (x + 1.) / 2.

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

ADAPTIVE_SOLVER = ["dopri5", "dopri8", "adaptive_heun", "bosh3"]
FIXER_SOLVER = ["euler", "rk4", "midpoint"]

def sample_from_model(model, x_0, args):
    if args.method in ADAPTIVE_SOLVER:
        options = {
            "dtype": torch.float64,
        }
    else:
        options = {
            "step_size": args.step_size,
            "perturb": args.perturb
        }
    if not args.compute_fid:
        model.count_nfe = True
    t = torch.tensor([1., 0.], device="cuda")
    fake_image = odeint(model, 
                        x_0, 
                        t, 
                        method=args.method, 
                        atol = args.atol, 
                        rtol = args.rtol,
                        adjoint_method=args.method,
                        adjoint_atol= args.atol,
                        adjoint_rtol= args.rtol,
                        options=options
                        )
    return fake_image


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def make_convolutional_sample(model, opt):
    log = dict()

    shape = [opt.batch_size,
             model.model.flowmatching_model.in_channels,
             model.model.flowmatching_model.image_size,
             model.model.flowmatching_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        x_0 = torch.randn(shape).to("cuda")
        sample = sample_from_model(model.model.flowmatching_model, x_0, opt)[-1]
        # sample, intermediates = model.sample(None, batch_size, is_training=True)
        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log

def run(model, logdir, opt):
    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))
    # path = logdir
    if model.cond_stage_model is None:
        print(f"Running unconditional sampling for {opt.n_samples} samples")
        for _ in trange(opt.n_samples // opt.batch_size, desc="Sampling Batches (unconditional)"):
            logs = make_convolutional_sample(model, opt)
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            print("NFE: {}".format(model.model.flowmatching_model.nfe))
            if n_saved >= opt.n_samples:
                print(f'Finish after generating {n_saved} samples')
                break
    else:
       raise NotImplementedError('Currently only sampling for unconditional models supported.')
    
    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample"):
    for k in logs:
        if k == key:
            batch = logs[key]
            for x in batch:
                img = custom_to_pil(x)
                imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                img.save(imgpath)
                n_saved += 1
            
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    parser.add_argument('--atol', type=float, default=1e-5, help='absolute tolerance error')
    parser.add_argument('--rtol', type=float, default=1e-5, help='absolute tolerance error')
    parser.add_argument('--method', type=str, default='dopri5', help='solver_method', choices=["dopri5", "dopri8", "adaptive_heun", "bosh3", "euler", "midpoint", "rk4"])
    parser.add_argument('--step_size', type=float, default=0.01, help='step_size')
    parser.add_argument('--perturb', action='store_true', default=False)
    
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    print(config)
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])
    model.eval()
    return model, global_step


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    logdir = "/".join(logdir.split("/")[:-1]) 
    base_configs = sorted(glob.glob(os.path.join(logdir, "configs", "*.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)
        
    model, global_step = load_model(config, ckpt)
    print(f"global step: {global_step}")
    print(75 * "=")
    
    logdir = os.path.join(logdir, "samples", f"{global_step:08}", now)
    imglogdir = os.path.join(logdir, "img")
    os.makedirs(imglogdir)
    print("logging to: {}".format(imglogdir))
    print(75 * "=")

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)


    run(model, imglogdir, opt)
    
    if opt.compute_fid:
        paths = [imglogdir, "pytorch_fid/celeba_256_stat.npy"]
        kwargs = {'batch_size': 50, 'device': "cuda", 'dims': 2048}
        fid = calculate_fid_given_paths(paths=paths, **kwargs)
        print('FID = {}'.format(fid))

    print("done.")
