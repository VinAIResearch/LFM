import torch

h = 32
w =32
b =4
mask = torch.ones(b, h, w)
                # # zeros will be filled in
mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
mask = mask[:, None, ...]