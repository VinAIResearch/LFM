import argparse
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
# from torchtoolbox.tools import summary

from models import create_network, get_flow_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser('flow-matching parameters')
    parser.add_argument('--generator', type=str, default="determ",
                        help='type of seed generator', choices=["dummy", "determ", "determ-indiv"])
    parser.add_argument('--seed', type=int, default=42,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--compute_nfe', action='store_true', default=False,
                            help='whether or not compute NFE')
    parser.add_argument('--measure_time', action='store_true', default=False,
                            help='wheter or not measure time')
    parser.add_argument('--epoch_id', type=int,default=1000)

    parser.add_argument('--model_type', type=str, default="adm",
                            help='model_type', choices=['adm', 'ncsn++', 'ddpm++', 'DiT-B/2', 'DiT-L/2', 'DiT-XL/2'])
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
                            help='channel of image')
    parser.add_argument('--n_sample', type=int, default=50000,
                            help='number of sampled images')
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
    parser.add_argument('--cfg_scale', type=float, default=1.,
                            help='Scale for classifier-free guidance')

    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)

    parser.add_argument('--pretrained_autoencoder_ckpt', type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument('--output_log', type=str, default="")
    
    #######################################
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy', help='directory to real images for FID computation')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--num_steps', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=200, help='sample generating batch size')
    
    # sampling argument
    parser.add_argument('--use_karras_samplers', action='store_true', default=False)
    parser.add_argument('--atol', type=float, default=1e-5, help='absolute tolerance error')
    parser.add_argument('--rtol', type=float, default=1e-5, help='absolute tolerance error')
    parser.add_argument('--method', type=str, default='dopri5', help='solver_method', choices=["dopri5", "dopri8", "adaptive_heun", "bosh3", 
        "euler", "midpoint", "rk4", "heun", "multistep", "stochastic", "dpm"])
    parser.add_argument('--step_size', type=float, default=0.01, help='step_size')
    parser.add_argument('--perturb', action='store_true', default=False)


    args = parser.parse_args()
    args.layout = False

    torch.manual_seed(42)
    device = 'cuda:0'

    # model = create_network(args).to(device)
    model = get_flow_model(args).to(device)
    model.eval()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params / 1024**2)
    
    print("Model size: {:.3f}MB".format(pytorch_total_params))

    # mem = 0.
    # for _ in range(300):
    #     x_t_1 = torch.randn(args.batch_size, args.num_in_channels, args.image_size//args.f, args.image_size//args.f).to(device)
    #     # t = torch.rand((args.batch_size,)).to(device)
    #     t = torch.tensor(1.0).to(device)

    #     x_0 = model(t, x_t_1)
    #     mem += torch.cuda.max_memory_allocated(device) / 2**30
    # print("Mem usage: {} (GB)".format(mem/300.))

    x = torch.randn(args.batch_size, args.num_in_channels, args.image_size//args.f, args.image_size//args.f).to(device)
    t = torch.ones(args.batch_size).to(device)

    flops = FlopCountAnalysis(model, (t, x))
    print(flop_count_table(flops))
    print(flops.total())

