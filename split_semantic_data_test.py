import os
import shutil
import torch
import torch.nn as nn

root_dir = "./dataset/CelebAMask-HQ/"
test_dir = "./celeba_test/"
os.makedirs(test_dir)
os.makedirs(os.path.join(test_dir, "img"))
os.makedirs(os.path.join(test_dir, "mask"))
def to_rgb(x):
    x = x.float()
    colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
    x = nn.functional.conv2d(x, weight=colorize)
    x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
    return x
for i in range(27000, 30000):
    shutil.copy(os.path.join(root_dir,"CelebA-HQ-img", "{}.jpg".format(i)), os.path.join(test_dir, "img", "{}.jpg".format(i)))
    shutil.copy(os.path.join(root_dir,"mask", "{}.png".format(i)), os.path.join(test_dir, "mask", "{}.png".format(i)))
    