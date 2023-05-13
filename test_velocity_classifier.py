# load the cifar checkpoint
import argparse
import torch
import torch.nn.functional as F
import os
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
from models.util import get_flow_model
import torchvision
from pytorch_fid.fid_score import calculate_fid_given_paths
import resnet


ADAPTIVE_SOLVER = ["dopri5", "dopri8", "adaptive_heun", "bosh3"]
FIXER_SOLVER = ["euler", "rk4", "midpoint"]

def get_cond(classifier, x, y, classifier_scale):
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in)
        log_probs = torch.log(logits)
        # print(log_probs)
        # exit()
        selected = log_probs[range(len(logits)), y.view(-1)]
        return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale

def get_sampler(flow_model):
    device = "cuda"
    n_steps = 1000
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
    
    def sampler_cond(x, y, classifier, classifier_scale):
        list_vec = []
        xs = []
        ones = torch.ones(x.shape[0], device=device)
        y  = torch.ones(x.shape[0], device=device)*y
        y = y.long()
        nfes = 0
        for n in range(n_steps):
            with torch.no_grad():
                vec = flow_model(ones * t[n], x)
            cond_eps = get_cond(classifier, vec, y, classifier_scale)
            print("cond {}".format(n), torch.max(cond_eps), torch.min(cond_eps))
            print("vec {}".format(n), torch.max(vec), torch.min(vec))
            vec = vec + cond_eps * (n_steps-n)/n_steps
            # list_vec.append(vec)
            nfes += 1
            h = (t[n + 1] - t[n])
            x = x + h * vec
            # xs.append(x)
        return x, nfes, list_vec, xs
    return sampler, sampler_cond

    

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


def sample_and_test(args):
    torch.manual_seed(42)
    device = 'cuda:0'
    
    if args.dataset == 'cifar10':
        real_img_dir = 'pytorch_fid/cifar10_train_stat.npy'
    elif args.dataset == 'celeba_256':
        real_img_dir = 'pytorch_fid/celeba_256_stat.npy'
    elif args.dataset == 'lsun':
        real_img_dir = 'pytorch_fid/lsun_church_stat.npy'
    else:
        real_img_dir = args.real_img_dir
    
    to_range_0_1 = lambda x: (x + 1.) / 2.

    classifier = resnet.resnet50().to(device)
    classifier_ckpt = torch.load('./saved_info/velocity_classifier/{}/test/model_275.pth'.format(args.dataset), map_location=device)
    print("Finish loading classifier")
    #loading weights from ddp in single gpu
    for key in list(classifier_ckpt.keys()):
        classifier_ckpt[key[7:]] = classifier_ckpt.pop(key)
    classifier.load_state_dict(classifier_ckpt)
    
    model =  get_flow_model(args).to(device)
    ckpt = torch.load('./saved_info/flow_matching/{}/{}/model_{}.pth'.format(args.dataset, args.exp, args.epoch_id), map_location=device)
    print("Finish loading model")
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    model.load_state_dict(ckpt)
    model.eval()
        
    iters_needed = 50000 //args.batch_size
    
    save_dir = "./generated_samples/{}".format(args.dataset)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if args.compute_fid:
        for i in range(iters_needed):
            with torch.no_grad():
                x_0 = torch.randn(args.batch_size, 3, args.image_size, args.image_size).to(device)
                fake_sample = sample_from_model(model, x_0, args)[-1]
                fake_sample = to_range_0_1(fake_sample)
                for j, x in enumerate(fake_sample):
                    index = i * args.batch_size + j 
                    torchvision.utils.save_image(x, './generated_samples/{}/{}.jpg'.format(args.dataset, index))
                print('generating batch ', i)
        
        paths = [save_dir, real_img_dir]
    
        kwargs = {'batch_size': 200, 'device': device, 'dims': 2048}
        fid = calculate_fid_given_paths(paths=paths, **kwargs)
        print('FID = {}'.format(fid))
    else:
        x_0 = torch.randn(args.batch_size, 3, args.image_size, args.image_size).to(device)
        sampler, sampler_cond = get_sampler(model)
        fake_sample, nfe, list_vec, xs = sampler_cond(x_0, 1, classifier, args.classifier_scale)
        print("NFE: {}".format(nfe))
        torchvision.utils.save_image(to_range_0_1(fake_sample), './samples_{}_{}.jpg'.format(args.dataset, args.classifier_scale))
        # os.makedirs("list_vec", exist_ok=True)
        # os.makedirs("list_image", exist_ok=True)
        # for idx, vec in enumerate(list_vec):
        #     torchvision.utils.save_image(to_range_0_1(vec), './list_vec/vec_{}.jpg'.format(idx))
        # torchvision.utils.save_image(to_range_0_1(fake_sample-x_0), './list_vec/gt_vec.jpg')
        # for idx, x in enumerate(xs):
        #     torchvision.utils.save_image(to_range_0_1(x), './list_image/img_{}.jpg'.format(idx))
        
        

    
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int,default=1000)

    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    parser.add_argument('--num_in_channels', type=int, default=3,
                            help='in channel image')
    parser.add_argument('--num_out_channels', type=int, default=3,
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
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--num_classes', type=int, default=None,
                            help='num classes')
    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)
    
    #######################################
    parser.add_argument('--exp', default='exp_1_OT', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy', help='directory to real images for FID computation')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--num_timesteps', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16, help='sample generating batch size')
    
    # sampling argument
    parser.add_argument('--atol', type=float, default=1e-5, help='absolute tolerance error')
    parser.add_argument('--classifier_scale', type=float, default=3, help='absolute tolerance error')
    parser.add_argument('--rtol', type=float, default=1e-5, help='absolute tolerance error')
    parser.add_argument('--method', type=str, default='euler', help='solver_method', choices=["dopri5", "dopri8", "adaptive_heun", "bosh3", "euler", "midpoint", "rk4"])
    parser.add_argument('--step_size', type=float, default=0.01, help='step_size')
    parser.add_argument('--perturb', action='store_true', default=False)
        



   
    args = parser.parse_args()
    
    sample_and_test(args)

