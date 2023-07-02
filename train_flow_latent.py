# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import shutil
import argparse
from functools import partial
from omegaconf import OmegaConf

import numpy as np
import torch
# faster training
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchdiffeq import odeint_adjoint as odeint
>>>>>>> origin/hao_dev
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.distributed as dist
from torch.multiprocessing import Process

from datasets_prep import get_dataset
from models import create_network
from EMA import EMA


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def get_weight(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


def sample_from_model(model, x_0):
    t = torch.tensor([1., 0.], dtype=x_0.dtype, device="cuda")
    fake_image = odeint(model, x_0, t, atol=1e-5, rtol=1e-5, adjoint_params=model.func.parameters())
    return fake_image

# def trace_df_dx_hutchinson(f, x, noise, no_autograd):
#     """
#     Hutchinson's trace estimator for Jacobian df/dx, O(1) call to autograd
#     """
#     if no_autograd:
#         # the following is compatible with checkpointing
#         torch.sum(f * noise).backward()
#         # torch.autograd.backward(tensors=[f], grad_tensors=[noise])
#         jvp = x.grad
#         trJ = torch.sum(jvp * noise, dim=[1, 2, 3])
#         x.grad = None
#     else:
#         jvp = torch.autograd.grad(f, x, noise, create_graph=False)[0]
#         trJ = torch.sum(jvp * noise, dim=[1, 2, 3])
#         # trJ = torch.einsum('bijk,bijk->b', jvp, noise)  # we could test if there's a speed difference in einsum vs sum
# 
#     return trJ

# def compute_ode_nll(self, dae, eps, ode_eps, ode_solver_tol, enable_autocast, no_autograd, num_samples, report_std):
#         """ calculates NLL based on ODE framework, assuming integration cutoff ode_eps """
#         # ODE solver starts consuming the CPU memory without this on large models
#         # https://github.com/scipy/scipy/issues/10070
#         gc.collect()

#         dae.eval()

#         def ode_func(t, state):
#             """ the ode function (including log probability integration for NLL calculation) """
#             global nfe_counter
#             nfe_counter = nfe_counter + 1

#             x = state[0].detach()
#             x.requires_grad_(True)
#             noise = torch.randn_like(x, device='cuda')  # could also use rademacher noise (sample_rademacher_like)
#             with torch.set_grad_enabled(True):
#                 with autocast(enabled=enable_autocast):
#                     variance = self.var(t=t)
#                     mixing_component = self.mixing_component(x_noisy=x, var_t=variance, t=t, enabled=dae.mixed_prediction)
#                     pred_params = dae(x=x, t=t)
#                     params = get_mixed_prediction(dae.mixed_prediction, pred_params, dae.mixing_logit, mixing_component)
#                     dx_dt = self.f(t=t) * x + 0.5 * self.g2(t=t) * params / torch.sqrt(variance)

#                 with autocast(enabled=False):
#                     dlogp_x_dt = -trace_df_dx_hutchinson(dx_dt, x, noise, no_autograd).view(x.shape[0], 1)

#             return (dx_dt, dlogp_x_dt)

#         # NFE counter
#         global nfe_counter

#         nll_all, nfe_all = [], []
#         for i in range(num_samples):
#             # integrated log probability
#             logp_diff_t0 = torch.zeros(eps.shape[0], 1, device='cuda')

#             nfe_counter = 0

#             # solve the ODE
#             x_t, logp_diff_t = odeint(
#                 ode_func,
#                 (eps, logp_diff_t0),
#                 torch.tensor([ode_eps, 1.0], device='cuda'),
#                 atol=ode_solver_tol,
#                 rtol=ode_solver_tol,
#                 method="scipy_solver",
#                 options={"solver": 'RK45'},
#             )
#             # last output values
#             x_t0, logp_diff_t0 = x_t[-1], logp_diff_t[-1]

#             # prior
#             if self.sde_type == 'vesde':
#                 logp_prior = torch.sum(util.distributions.log_p_var_normal(x_t0, var=self.sigma2_max), dim=[1, 2, 3])
#             else:
#                 logp_prior = torch.sum(util.distributions.log_p_standard_normal(x_t0), dim=[1, 2, 3])

#             log_likelihood = logp_prior - logp_diff_t0.view(-1)

#             nll_all.append(-log_likelihood)
#             nfe_all.append(nfe_counter)

#         nfe_mean = np.mean(nfe_all)
#         nll_all = torch.stack(nll_all, dim=1)
#         nll_mean = torch.mean(nll_all, dim=1)
#         if num_samples > 1 and report_std:
#             nll_stddev = torch.std(nll_all, dim=1)
#             nll_stddev_batch = torch.mean(nll_stddev)
#             nll_stderror_batch = nll_stddev_batch / np.sqrt(num_samples)
#         else:
#             nll_stddev_batch = None
#             nll_stderror_batch = None
#         return nll_mean, nfe_mean, nll_stddev_batch, nll_stderror_batch

#%%
def train(rank, gpu, args):
    from diffusers.models import AutoencoderKL
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    device = torch.device('cuda:{}'.format(gpu))
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32

    batch_size = args.batch_size

    dataset = get_dataset(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last = True)

    model = create_network(args).to(device, dtype=dtype)
    if args.use_grad_checkpointing and "DiT" in args.model_type:
        model.set_gradient_checkpointing()

    first_stage_model = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device, dtype=dtype)
    first_stage_model = first_stage_model.eval()
    first_stage_model.train = False
    for param in first_stage_model.parameters():
        param.requires_grad = False

    print('AutoKL size: {:.3f}MB'.format(get_weight(first_stage_model)))
    print('FM size: {:.3f}MB'.format(get_weight(model)))

    broadcast_params(model.parameters())

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    if args.use_ema:
        optimizer = EMA(optimizer, ema_decay=args.ema_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-5)

    #ddp
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=False)

    exp = args.exp
    parent_dir = "./saved_info/latent_flow/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            config_dict = vars(args)
            OmegaConf.save(config_dict, os.path.join(exp_path, "config.yaml"))
    print("Exp path:", exp_path)

    if args.resume or os.path.exists(os.path.join(exp_path, 'content.pth')):
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        model.load_state_dict(checkpoint['model_dict'])
        # load G
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint["global_step"]

        print("=> resume checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        del checkpoint

    elif args.model_ckpt and os.path.exists(os.path.join(exp_path, args.model_ckpt)):
        checkpoint_file = os.path.join(exp_path, args.model_ckpt)
        checkpoint = torch.load(checkpoint_file, map_location=device)
        epoch = int(args.model_ckpt.split("_")[-1][:-4])
        init_epoch = epoch
        model.load_state_dict(checkpoint)
        global_step = 0

        print("=> loaded checkpoint (epoch {})"
                  .format(epoch))
        del checkpoint
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    use_label = True if "imagenet" in args.dataset else False
    is_latent_data = True if "latent" in args.dataset else False
    for epoch in range(init_epoch, args.num_epoch+1):
        train_sampler.set_epoch(epoch)

        for iteration, (x, y) in enumerate(data_loader):
            x_0 = x.to(device, dtype=dtype, non_blocking=True)
            y = None if not use_label else y.to(device, non_blocking=True)
            model.zero_grad()
            if is_latent_data:
                z_0 = x_0 * args.scale_factor
            else:
                z_0 = first_stage_model.encode(x_0).latent_dist.sample().mul_(args.scale_factor)
            #sample t
            t = torch.rand((z_0.size(0),), dtype=dtype, device=device)
            t = t.view(-1, 1, 1, 1)
            z_1 = torch.randn_like(z_0)
            # corrected notation: 1 is real noise, 0 is real data
            v_t = (1 - t) * z_0 + (1e-5 + (1 - 1e-5) * t) * z_1
            u = (1 - 1e-5) * z_1 - z_0
            # alternative notation (similar to flow matching): 1 is data, 0 is real noise
            # v_t = (1 - (1 - 1e-5) * t) * z_0 + t * z_1
            # u = z_1 - (1 - 1e-5) * z_0
            v = model(t.squeeze(), v_t, y)
            loss = F.mse_loss(v, u)
            loss.backward()
            optimizer.step()
            global_step += 1
            if iteration % 100 == 0:
                if rank == 0:
                    print('epoch {} iteration{}, Loss: {}'.format(epoch,iteration, loss.item()))

        if not args.no_lr_decay:
            scheduler.step()

        if rank == 0:
            if epoch % args.plot_every == 0:

                with torch.no_grad():
                    rand = torch.randn_like(z_0)[:4]
                    if y is not None:
                        y = y[:4]
                    sample_model = partial(model, y=y)
                    # sample_func = lambda t, x: model(t, x, y=y)
                    fake_sample = sample_from_model(sample_model, rand)[-1]
                    fake_image = first_stage_model.decode(fake_sample / args.scale_factor).sample
                # torchvision.utils.save_image(fake_sample, os.path.join(exp_path, 'sample_epoch_{}.png'.format(epoch)), normalize=True, value_range=(-1, 1))
                torchvision.utils.save_image(fake_image, os.path.join(exp_path, 'image_epoch_{}.png'.format(epoch)), normalize=True, value_range=(-1, 1))
                print("Finish sampling")

            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
                    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                               'model_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                               'scheduler': scheduler.state_dict()}

                    torch.save(content, os.path.join(exp_path, 'content.pth'))

            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizer.swap_parameters_with_ema(store_params_in_ema=True)

                torch.save(model.state_dict(), os.path.join(exp_path, 'model_{}.pth'.format(epoch)))
                if args.use_ema:
                    optimizer.swap_parameters_with_ema(store_params_in_ema=True)



def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()


#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')

    parser.add_argument('--resume', action='store_true',default=False)
    parser.add_argument('--model_ckpt', type=str, default=None,
                            help="Model ckpt to init from")

    parser.add_argument('--model_type', type=str, default="adm",
                            help='model_type', choices=['adm', 'ncsn++', 'ddpm++', 'DiT-B/2', 'DiT-L/2', 'DiT-L/4', 'DiT-XL/2'])
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    parser.add_argument('--f', type=int, default=8,
                            help='downsample rate of input image by the autoencoder')
    parser.add_argument('--scale_factor', type=float, default=0.18215,
                            help='size of image')
    parser.add_argument('--num_in_channels', type=int, default=3,
                            help='in channel image')
    parser.add_argument('--num_out_channels', type=int, default=3,
                            help='in channel image')
    parser.add_argument('--nf', type=int, default=256,
                            help='channel of model')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', nargs='+', type=int, default=(16,),
                            help='resolution of applying attention')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=(1,1,2,2,4,4),
                            help='channel mult')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--label_dim', type=int, default=0,
                            help='label dimension, 0 if unconditional')
    parser.add_argument('--augment_dim', type=int, default=0,
                            help='dimension of augmented label, 0 if not used')
    parser.add_argument('--num_classes', type=int, default=None,
                            help='num classes')
    parser.add_argument('--label_dropout', type=float, default=0.,
                            help='Dropout probability of class labels for classifier-free guidance')

    # Original ADM
    parser.add_argument('--layout', action='store_true')
    parser.add_argument('--use_origin_adm', action='store_true')
    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument("--resamp_with_conv", type=bool, default=True)
    parser.add_argument('--num_heads', type=int, default=4,
                            help='number of head')
    parser.add_argument('--num_head_upsample', type=int, default=-1,
                            help='number of head upsample')
    parser.add_argument('--num_head_channels', type=int, default=-1,
                            help='number of head channels')

    parser.add_argument('--pretrained_autoencoder_ckpt', type=str, default="stabilityai/sd-vae-ft-mse")

    # training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--datadir', default='./data')
    parser.add_argument('--num_timesteps', type=int, default=200)
    parser.add_argument('--use_bf16', action='store_true', default=False)
    parser.add_argument('--use_grad_checkpointing', action='store_true', default=False,
        help="Enable gradient checkpointing for mem saving")

    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)

    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate g')

    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--no_lr_decay',action='store_true', default=False)

    parser.add_argument('--use_ema', action='store_true', default=False,
                            help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')


    parser.add_argument('--save_content', action='store_true', default=False)
    parser.add_argument('--save_content_every', type=int, default=10, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=25, help='save ckpt every x epochs')
    parser.add_argument('--plot_every', type=int, default=5, help='plot every x epochs')

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

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print('starting in debug mode')

        init_processes(0, size, train, args)
