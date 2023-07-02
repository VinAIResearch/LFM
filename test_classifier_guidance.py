import argparse
import torch
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from torchdiffeq import odeint_adjoint as odeint
import torchvision

import math

import torch.distributed as dist
from torch.multiprocessing import Process

from pytorch_fid.fid_score import calculate_fid_given_paths
from models.encoder_classifier import create_classifier, classifier_defaults

from sampler.karras_sample import karras_sample
from sampler.random_util import get_generator

from ddp_utils import init_processes
from models import create_network
# from models.util import get_flow_model
from test_flow_latent import NFECount

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

ADAPTIVE_SOLVER = ["dopri5", "dopri8", "adaptive_heun", "bosh3"]
FIXER_SOLVER = ["euler", "rk4", "midpoint", "stochastic"]


def get_cond(classifier, x, s, y, scale, **kwargs):
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        s_ = torch.tensor(s).expand(x.size(0),).to("cuda")
        logits = classifier(x_in, s_)
        log_probs = torch.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        return 2 * torch.autograd.grad(selected.sum(), x_in)[0] * scale * s/(1 + 1e-4 - s)

def get_sampler(flow_model, n_steps, device = "cuda"):
    t_final = 0.
    t_start = 1.
    t = torch.linspace(t_start, t_final, n_steps + 1, device=device)
    
    def sampler(x):
        list_vec = []
        xs = []
        ones = torch.ones(x.shape[0], device=device)
        nfes = 0
        for n in range(n_steps):
            vec = flow_model(ones * t[n], x)
            # list_vec.append(vec)
            nfes += 1
            h = (t[n + 1] - t[n])
            x = x + h * vec
            # xs.append(x)
        return x, nfes, list_vec, xs
    
    def sampler_cond(x, y, classifier, scale):
        list_vec = []
        xs = []
        ones = torch.ones(x.shape[0], device=device)
        y  = torch.randint(0, 1000, (x.shape[0],), device=device)
        y = y.long()
        nfes = 0
        for n in range(n_steps):
            with torch.no_grad():
                vec = flow_model(t[n], x, y)
            cond_vec = get_cond(classifier, x, 1-t[n], y, scale)
            # print("cond {}".format(n), torch.max(cond_eps), torch.min(cond_eps))
            # print("vec {}".format(n), torch.max(vec), torch.min(vec))
            vec = vec + cond_vec
            # list_vec.append(vec)
            nfes += 1
            h = (t[n + 1] - t[n])
            x = x + h*vec
            # xs.append(x)
        return x, nfes, list_vec, xs
    return sampler, sampler_cond


def sample_from_model(model, x_0, model_kwargs, classifier, args):
    if args.method in ADAPTIVE_SOLVER:
        options = {
            "dtype": torch.float64,
        }
    else:
        options = {
            "step_size": args.step_size,
            "perturb": args.perturb
        }
    if args.compute_nfe:
        # model.count_nfe = True
        model = NFECount(model).to(x_0.device) # count wrapper

    t = torch.tensor([1., 0.], device="cuda")

    def denoiser(t, x_0):
        if args.cfg_scale > 1.:
            vec = model.forward_with_cfg(t, x_0, **model_kwargs)
        else:
            vec = model(t, x_0, **model_kwargs)
        # vec = model(t, x_0)
        cond_vec = get_cond(classifier, x_0, 1.-t, **model_kwargs)
        return vec + cond_vec

    fake_image = odeint(denoiser, 
                        x_0, 
                        t, 
                        method=args.method, 
                        atol = args.atol, 
                        rtol = args.rtol,
                        adjoint_method=args.method,
                        adjoint_atol= args.atol,
                        adjoint_rtol= args.rtol,
                        options=options,
                        adjoint_params=model.parameters(),
                        )
    if args.compute_nfe:
        return fake_image, model.nfe
    return fake_image


def sample_from_model2(model, x, model_kwargs, generator, classifier, args):
    sample = karras_sample(
            model,
            x,
            steps=args.num_steps,
            model_kwargs=model_kwargs,
            device=x.device,
            clip_denoised=False,
            sigma_min=1e-5,
            sigma_max=1.0,
            s_tmin=0.,
            s_tmax=1.0,
            s_churn=0.,
            sampler=args.method,
            rho=1.0,
            ts=range(0, args.num_steps, 15),
            generator=generator,
            classifier=classifier,
            cond_func=get_cond,
        )
    return sample


# def get_sampler(flow_model, n_steps, device = "cuda"):
#     t_final = 0.
#     t_start = 1.
#     t = torch.linspace(t_start, t_final, n_steps + 1, device=device)
    
#     def sampler(x):
#         list_vec = []
#         xs = []
#         ones = torch.ones(x.shape[0], device=device)
#         nfes = 0
#         for n in range(n_steps):
#             vec = flow_model(ones * t[n], x)
#             # list_vec.append(vec)
#             nfes += 1
#             h = (t[n + 1] - t[n])
#             x = x + h * vec
#             # xs.append(x)
#         return x, nfes, list_vec, xs
    
#     def sampler_cond(x, y, classifier, scale):
#         list_vec = []
#         xs = []
#         ones = torch.ones(x.shape[0], device=device)
#         y  = torch.ones(x.shape[0], device=device)*y
#         y = y.long()
#         nfes = 0
#         for n in range(n_steps):
#             s = n/n_steps
#             with torch.no_grad():
#                 vec = flow_model(ones * t[n], x)
#             cond_vec = get_cond(classifier, x, s, y, scale)
#             # print("cond {}".format(n), torch.max(cond_eps), torch.min(cond_eps))
#             # print("vec {}".format(n), torch.max(vec), torch.min(vec))
#             vec = vec + cond_vec
#             # list_vec.append(vec)
#             nfes += 1
#             h = (t[n + 1] - t[n])
#             x = x + h * vec
#             # xs.append(x)
#         return x, nfes, list_vec, xs
#     return sampler, sampler_cond


def sample_and_test(rank, gpu, args):
    from diffusers.models import AutoencoderKL
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    seed = args.seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda:{}'.format(gpu))
>>>>>>> origin/hao_dev
    
    if args.dataset == 'cifar10':
        real_img_dir = 'pytorch_fid/cifar10_train_stat.npy'
    elif args.dataset == 'celeba_256':
        real_img_dir = 'pytorch_fid/celebahq_stat.npy'
    elif args.dataset == 'lsun_church':
        real_img_dir = 'pytorch_fid/lsun_church_stat.npy'
    elif args.dataset == "ffhq_256":
        real_img_dir = 'pytorch_fid/ffhq_stat.npy'
    elif args.dataset == "lsun_bedroom":
        real_img_dir = 'pytorch_fid/lsun_bedroom_stat.npy'
    elif args.dataset in ["latent_imagenet_256", "imagenet_256"]:
        real_img_dir = 'pytorch_fid/imagenet_stat.npy'
    else:
        real_img_dir = args.real_img_dir
    
    to_range_0_1 = lambda x: (x + 1.) / 2.
    
    classifier = create_classifier(
        **args_to_dict(args, classifier_defaults().keys())
    ).to(device)
    # classifier_ckpt = torch.load('./saved_info/classifier_guidance//{}/{}'.format(args.dataset, args.classifier_ckpt), map_location=device)
    
    classifier_ckpt = torch.load(args.classifier_ckpt, map_location=device)
    print("Finish loading classifier")
    #loading weights from ddp in single gpu
    for key in list(classifier_ckpt.keys()):
        classifier_ckpt[key[7:]] = classifier_ckpt.pop(key)
    classifier.load_state_dict(classifier_ckpt)
    # model =  get_flow_model(args).to(device)
    model = create_network(args).to(device)
    ckpt = torch.load('./saved_info/latent_flow/{}/{}/model_{}.pth'.format(args.dataset, args.exp, args.epoch_id), map_location=device)
    print("Finish loading model")
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    model.load_state_dict(ckpt)
    model.eval()

    first_stage_model = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device)
        
    iters_needed = args.n_sample //args.batch_size
    
    save_dir = "./generated_samples/{}/cls_exp{}_ep{}_m{}".format(args.dataset, args.exp, args.epoch_id, args.method)
    if args.method in FIXER_SOLVER:
        save_dir += "_s{}".format(args.num_steps)
    save_dir += "_scale{}".format(args.scale)
    if args.cfg_scale > 1.:
        save_dir += "_cfg{}".format(args.cfg_scale)
    # save_dir = "./generated_samples/{}".format(args.dataset)
    
    if rank == 0 and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # seed generator
    #### seed should be aligned with rank 
    #### as the same seed can cause identical generation on other gpus
    generator = get_generator(args.generator, args.n_sample, seed)
    
    if args.compute_fid:
        print("Compute fid")
        dist.barrier()

        # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
        n = args.batch_size
        global_batch_size = n * args.world_size
        total_samples = int(math.ceil(args.n_sample / global_batch_size) * global_batch_size)
        if rank == 0:
            print(f"Total number of images that will be sampled: {total_samples}")
        assert total_samples % args.world_size == 0, "total_samples must be divisible by world_size"
        samples_needed_this_gpu = int(total_samples // args.world_size)
        iters_needed = int(samples_needed_this_gpu // n)
        pbar = range(iters_needed)
        pbar = tqdm(pbar) if rank == 0 else pbar
        total = 0

        for i in pbar:
            with torch.no_grad():
                x = generator.randn(args.batch_size, args.num_in_channels, args.classifier_image_size, args.classifier_image_size).to(device)
                y = generator.randint(0, args.num_classes, (args.batch_size,), device=device).long()
                # Setup classifier-free guidance:
                if args.cfg_scale > 1.:
                    x = torch.cat([x, x], 0)
                    y_null = torch.tensor([args.num_classes] * y.size(0), device=device).long() if "DiT" in args.model_type else torch.zeros_like(y)
                    y = torch.cat([y, y_null], 0)
                
                _, sampler_cond = get_sampler(model, n_steps=int(1/args.step_size), device=device)
                fake_sample, nfe, list_vec, xs = sampler_cond(x, y, classifier, args.scale)

                # model_kwargs = {'y': y, 'cfg_scale': args.cfg_scale, 'scale': args.scale}
                # if not args.use_karras_samplers:
                #     fake_sample = sample_from_model(model, x, model_kwargs, classifier, args)[-1]
                # else:
                #     fake_sample = sample_from_model2(model, x, model_kwargs, generator, classifier, args)

                fake_sample = first_stage_model.decode(fake_sample / args.scale_factor).sample
                fake_sample = torch.clamp(to_range_0_1(fake_sample), 0, 1)

                for j, x in enumerate(fake_sample):
                    index = j * args.world_size + rank + total
                    torchvision.utils.save_image(x, '{}/{}.jpg'.format(save_dir, index))
                if rank == 0:
                    print('generating batch ', i)
                total += global_batch_size
        

        # make sure all processes have finished
        dist.barrier()
        if rank == 0:
            paths = [save_dir, real_img_dir]
            kwargs = {'batch_size': 200, 'device': device, 'dims': 2048}
            fid = calculate_fid_given_paths(paths=paths, **kwargs)
            print('FID = {}'.format(fid))
            with open(args.output_log, "a") as f:
                f.write('Epoch = {}, FID = {}, scale = {}, cfg = {} \n'.format(args.epoch_id, fid, args.scale, args.cfg_scale))
        dist.barrier()
        dist.destroy_process_group()
    else:
        print("Inference")
        x = generator.randn(args.batch_size, args.num_in_channels, args.classifier_image_size, args.classifier_image_size).to(device)
        # y = 0 * torch.ones(args.batch_size, device=device).long()
        y = generator.randint(0, args.num_classes, (args.batch_size,), device=device).long()
        # Setup classifier-free guidance:
        if args.cfg_scale > 1.:
            x = torch.cat([x, x], 0)
            y_null = torch.tensor([args.num_classes] * y.size(0), device=device).long() if "DiT" in args.model_type else torch.zeros_like(y)
            y = torch.cat([y, y_null], 0)

        _, sampler_cond = get_sampler(model, n_steps=int(1/args.step_size), device=device)
        fake_sample, nfe, list_vec, xs = sampler_cond(x, y, classifier, args.scale)
        # print("NFE: {}".format(nfe))

        # model_kwargs = {'y': y, 'cfg_scale': args.cfg_scale, 'scale': args.scale}
        # if not args.use_karras_samplers:
        #     fake_sample = sample_from_model(model, x, model_kwargs, classifier, args)[-1]
        # else:
        #     fake_sample = sample_from_model2(model, x, model_kwargs, generator, classifier, args)
        fake_image = first_stage_model.decode(fake_sample / args.scale_factor).sample
        fake_image = torch.clamp(to_range_0_1(fake_image), 0, 1)

        if not args.use_karras_samplers:
            save_path = './cls_samples_{}_{}_scale{}_cfg{}.jpg'.format(args.dataset, args.method, args.scale, args.cfg_scale)
        else:
            save_path = './cls_samples_{}_{}_{}_scale{}_cfg{}.jpg'.format(args.dataset, args.method, args.num_steps, args.scale, args.cfg_scale)

        if args.cfg_scale > 1.:
            fake_sample, _ = fake_sample.chunk(2, dim=0)  # Remove null class samples
        
        torchvision.utils.save_image(fake_image, save_path)
        print("Samples are save at '{}".format(save_path))

        # os.makedirs("list_vec", exist_ok=True)
        # os.makedirs("list_image", exist_ok=True)
        # for idx, vec in enumerate(list_vec):
        #     torchvision.utils.save_image(to_range_0_1(vec), './list_vec/vec_{}.jpg'.format(idx))
        # torchvision.utils.save_image(to_range_0_1(fake_sample-x_0), './list_vec/gt_vec.jpg')
        # for idx, x in enumerate(xs):
        #     torchvision.utils.save_image(to_range_0_1(x), './list_image/img_{}.jpg'.format(idx))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--generator', type=str, default="determ",
                        help='type of seed generator', choices=["dummy", "determ", "determ-indiv"])
    parser.add_argument('--seed', type=int, default=42,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--compute_nfe', action='store_true', default=False,
                            help='whether or not compute NFE')
    parser.add_argument('--epoch_id', type=int,default=1000)
    parser.add_argument('--n_sample', type=int, default=50000,
                            help='number of sampled images')

    parser.add_argument('--model_type', type=str, default="adm",
                            help='model_type', choices=['adm', 'ncsn++', 'ddpm++', 'DiT-B/2', 'DiT-L/2', 'DiT-XL/2'])
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    parser.add_argument('--f', type=int, default=8,
                            help='downsample rate of input image by the autoencoder')
    parser.add_argument('--scale_factor', type=float, default=0.18215,
                            help='size of image')
    parser.add_argument('--num_in_channels', type=int, default=4,
                            help='in channel image')
    parser.add_argument('--num_out_channels', type=int, default=4,
>>>>>>> origin/hao_dev
                            help='in channel image')
    parser.add_argument('--nf', type=int, default=256,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument("--resamp_with_conv", type=bool, default=True)
    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--num_heads', type=int, default=4,
                            help='number of head')
    parser.add_argument('--num_head_upsample', type=int, default=-1,
                            help='number of head upsample')
    parser.add_argument('--num_head_channels', type=int, default=-1,
                            help='number of head channels')
    parser.add_argument('--attn_resolutions', nargs='+', type=int, default=(16,),
                            help='resolution of applying attention')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=(1,2,2,2),
                            help='channel mult')
    parser.add_argument('--label_dim', type=int, default=0,
                            help='label dimension, 0 if unconditional')
    parser.add_argument('--augment_dim', type=int, default=0,
                            help='dimension of augmented label, 0 if not used')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--num_classes', type=int, default=None,
                            help='num classes')
    parser.add_argument('--label_dropout', type=float, default=0.,
                            help='Dropout probability of class labels for classifier-free guidance')

    # parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    # parser.add_argument("--resblock_updown", type=bool, default=False)
    # parser.add_argument("--use_new_attention_order", type=bool, default=False)

    parser.add_argument('--pretrained_autoencoder_ckpt', type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument('--output_log', type=str, default="")
    
    parser.add_argument('--exp', default='exp_1_OT', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy', help='directory to real images for FID computation')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--num_timesteps', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16, help='sample generating batch size')
    parser.add_argument('--classifier_image_size', type=int, default=32, help='num of resblock')    
    parser.add_argument('--classifier_depth', type=int, default=3, help='num of resblock')
    parser.add_argument('--classifier_width', type=int, default=192, help='num of resblock')
    parser.add_argument('--classifier_pool', type=str, default="attention", help='num of resblock')
    parser.add_argument('--classifier_resblock_updown', type=bool, default=True, help='num of resblock')
    parser.add_argument('--classifier_use_scale_shift_norm', type=bool, default=True, help='num of resblock')
    parser.add_argument('--classifier_use_fp16', type=bool, default=False, help='num of resblock')
    parser.add_argument('--classifier_attention_resolutions', type=str, default="8,4", help='num of resblock')
    parser.add_argument('--classifier_ckpt', type=str, default="", help='Classifier checkpoint ')
    
    # sampling argument
    parser.add_argument('--use_karras_samplers', action='store_true', default=False)
    parser.add_argument('--num_steps', type=int, default=40)
    parser.add_argument('--atol', type=float, default=1e-5, help='absolute tolerance error')
    parser.add_argument('--rtol', type=float, default=1e-5, help='absolute tolerance error')
    parser.add_argument('--method', type=str, default='dopri5', help='solver_method', choices=["dopri5", "dopri8", "adaptive_heun", "bosh3", 
        "euler", "midpoint", "rk4", "heun", "multistep", "stochastic", "dpm"])
    parser.add_argument('--step_size', type=float, default=0.01, help='step_size')
    parser.add_argument('--perturb', action='store_true', default=False)
    parser.add_argument('--cfg_scale', type=float, default=1.,
                            help='Scale for classifier-free guidance')
    parser.add_argument('--scale', type=float, default=-1.5,
                            help='Scale for classifier-guided guidance')
        
    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--master_port', type=str, default='6000',
                        help='port for master')

    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1 and args.compute_fid:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, sample_and_test, args))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
    else:
        print('starting in debug mode')
        
        init_processes(0, size, sample_and_test, args)
>>>>>>> origin/hao_dev
