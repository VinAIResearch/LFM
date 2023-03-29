#!/usr/bin/env python
import os, sys
# sys.path.append("~/workspace/quandm7/CNF_FM")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.datasets import make_moons
from torch import Tensor
from torch.distributions import Normal
from tqdm import tqdm
from typing import *

from zuko.utils import odeint


class MLP(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int] = [64, 64],
    ):
        layers = []

        for a, b in zip(
            [in_features] + hidden_features,
            hidden_features + [out_features],
        ):
            layers.extend([nn.Linear(a, b), nn.ELU()])

        super().__init__(*layers[:-1])



class CNF(nn.Module):
    def __init__(
        self,
        features: int,
        frequencies: int = 3,
        **kwargs,
    ):
        super().__init__()

        self.net = MLP(2 * frequencies + features, features, **kwargs)

        self.register_buffer('frequencies', 2 ** torch.arange(frequencies) * torch.pi)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        t = self.frequencies * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        t = t.expand(*x.shape[:-1], -1)

        return self.net(torch.cat((t, x), dim=-1))

    def encode(self, x: Tensor) -> Tensor:
        return odeint(self, x, 0.0, 1.0, phi=self.parameters())

    def decode(self, z: Tensor) -> Tensor:
        return odeint(self, z, 1.0, 0.0, phi=self.parameters())

    def log_prob(self, x: Tensor) -> Tensor:
        I = torch.eye(x.shape[-1]).to(x)
        I = I.expand(x.shape + x.shape[-1:]).movedim(-1, 0)

        def augmented(t: Tensor, x: Tensor, ladj: Tensor) -> Tensor:
            with torch.enable_grad():
                x = x.requires_grad_()
                dx = self(t, x)

            jacobian = torch.autograd.grad(dx, x, I, is_grads_batched=True, create_graph=True)[0]
            trace = torch.einsum('i...i', jacobian)

            return dx, trace * 1e-2

        ladj = torch.zeros_like(x[..., 0])
        z, ladj = odeint(augmented, (x, ladj), 0.0, 1.0, phi=self.parameters())

        return Normal(0.0, z.new_tensor(1.0)).log_prob(z).sum(dim=-1) + ladj * 1e2


class FlowMatchingLoss(nn.Module):
    def __init__(self, v: nn.Module):
        super().__init__()

        self.v = v

    def forward(self, x: Tensor) -> Tensor:
        t = torch.rand_like(x[..., 0]).unsqueeze(-1)
        z = torch.randn_like(x)
        y = (1 - t) * x + (1e-15 + (1 - 1e-15) * t) * z
        u = (1 - 1e-15) * z - x + 0.1 * (z-x)/t
        
        t, y, u = t.to("cuda"), y.to("cuda"), u.to("cuda")

        return (self.v(t.squeeze(-1), y) - u).square().mean()


if __name__ == '__main__':
    flow = CNF(2, hidden_features=[256] * 3).to("cuda")

    # Training
    loss = FlowMatchingLoss(flow)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-3)

    data, _ = make_moons(4096, noise=0.05)
    data = torch.from_numpy(data).float()

    for epoch in tqdm(range(4096), ncols=88):
        subset = torch.randint(0, len(data), (256,))
        x = data[subset].to("cuda")

        optimizer.zero_grad()
        loss(x).backward()
        optimizer.step()

    # Sampling
    with torch.no_grad():
        z = torch.randn(4096, 2).to("cuda")
        x = flow.decode(z).to("cpu").numpy()

    plt.hist2d(x[:, 0], x[:, 1], bins=64)
    plt.savefig('moons.png', dpi=300)

    # Log-likelihood
    with torch.no_grad():
        log_p = flow.log_prob(data.to("cuda"))

    print(-log_p.mean())