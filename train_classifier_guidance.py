# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import torch
import numpy as np
from models.encoder_classifier import create_classifier, classifier_defaults
from torchdiffeq import odeint_adjoint as odeint
import os
from datasets_prep.latent_datasets import LatentDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets_prep import get_dataset
from torch.multiprocessing import Process
import torch.distributed as dist
import shutil
import time
import torchvision.transforms as transforms
from tqdm import tqdm



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))
            
def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)

def sample_from_model(model, x_0):
    t = torch.tensor([1., 0.], device="cuda")
    fake_image = odeint(model, x_0, t, atol=1e-5, rtol=1e-5)
    return fake_image

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

#%%
def train(rank, gpu, args):
    
    from EMA import EMA
    
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))
    
    batch_size = args.batch_size
    
    
    dataset = LatentDataset(root=args.data_dir, train=True, transform=transforms.Compose([
                        transforms.RandomHorizontalFlip()]))
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
    
    print(args_to_dict(args, classifier_defaults().keys()))
    
    model = create_classifier(
        **args_to_dict(args, classifier_defaults().keys())
    ).to(device)
    
    broadcast_params(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas = (args.beta1, args.beta2))
    
    if args.use_ema:
        optimizer = EMA(optimizer, ema_decay=args.ema_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-5)
    
    #ddp
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    exp = args.exp
    parent_dir = "./saved_info/classifier_guidance/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
    
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        model.load_state_dict(checkpoint['model_dict'])
        # load G
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
    else:
        epoch, init_epoch = 0, 0
    
    
    for epoch in range(init_epoch, args.num_epoch+1):
        train_sampler.set_epoch(epoch)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        for (x, y) in tqdm(data_loader, desc="Epoch {}".format(epoch)):
            data_time.update(time.time() - end)
            x_1 = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            model.zero_grad()
            #sample t
            t = torch.rand((x_1.size(0),) , device=device)
            t = t.view(-1, 1, 1, 1)
            x_0 = torch.randn_like(x_1)
            x_t = (1 - t) * x_1 + (1e-5 + (1 - 1e-5) * t) * x_0
            
            # compute loss
            logits = model(x_t, t.squeeze())
            loss = F.cross_entropy(logits, y)
            losses.update(loss.item(), batch_size)
            acc1, acc5 = accuracy(logits, y, topk=(1, 5))
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            
            loss.backward()
            optimizer.step()
            
                    
        
        if not args.no_lr_decay:
            scheduler.step()
        
        if rank == 0:
            print('epoch {}, Loss: {}, Top1: {}, Top5: {}, Data Time: {}'.format(epoch, loss.item(), top1.avg, top5.avg, data_time.avg))
            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
                    content = {'epoch': epoch + 1, 'args': args,
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
    os.environ['MASTER_PORT'] = '6021'
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
    
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    parser.add_argument('--num_in_channels', type=int, default=3,
                            help='in channel image')
    
    parser.add_argument('--exp', default='test', help='name of experiment')
    parser.add_argument('--data_dir', default='test', help='name of experiment')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')

    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    
    parser.add_argument('--classifier_depth', type=int, default=2, help='num of resblock')
    parser.add_argument('--classifier_width', type=int, default=128, help='num of resblock')
    parser.add_argument('--classifier_pool', type=str, default="attention", help='num of resblock')
    parser.add_argument('--classifier_resblock_updown', type=bool, default=True, help='num of resblock')
    parser.add_argument('--classifier_use_scale_shift_norm', type=bool, default=True, help='num of resblock')
    parser.add_argument('--classifier_use_fp16', type=bool, default=False, help='num of resblock')
    parser.add_argument('--classifier_attention_resolutions', type=str, default="8,4", help='num of resblock')
    
        
    parser.add_argument('--num_epoch', type=int, default=500)

    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate g')
    
    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--no_lr_decay',action='store_true', default=False)
    
    parser.add_argument('--use_ema', action='store_true', default=False,
                            help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    

    parser.add_argument('--save_content', action='store_true',default=False)
    parser.add_argument('--save_content_every', type=int, default=10, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=25, help='save ckpt every x epochs')
   
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