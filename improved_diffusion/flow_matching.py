import enum
import math

import numpy as np
import torch as th
from zuko.zuko.utils import odeint


def get_named_beta_schedule(schedule_name, num_flow_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_flow_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_flow_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_flow_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
    

def betas_for_alpha_bar(num_flow_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_flow_timesteps):
        t1 = i / num_flow_timesteps
        t2 = (i + 1) / num_flow_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class FlowType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    OT = enum.auto()
    DIFFFUSION = enum.auto()
    

class FlowMatching:
    def __init__(
        self,
        num_timesteps,
        flow_type,
        sigma_min=1e-4
    ):
        self.flow_type=flow_type
        self.num_timesteps = num_timesteps
        self.sigma_min = th.tensor(sigma_min)
        
    def get_flow_t(self, x1, x0, t):
        if self.flow_type == FlowType.OT:
            return (1 - (1 - self.sigma_min))*x0 + t*x1
        else:
            raise TypeError
        
    def get_u_t(self, x1, x0, t):
        if self.flow_type == FlowType.OT:
            return x1 - (1-self.sigma_min.to("cuda"))*x0
        else:
            raise TypeError
        
    def encode(self, model, x1):
        return odeint(model, x1, 0.0, 1.0, phi=model.parameters())
    
    def decode(self, model, x0):
        return odeint(model, x0, 1.0, 0.0, phi=model.parameters())
    
    
    def training_losses(self, model, x1, t, model_kwargs=None, noise=None):
        x0 = th.rand_like(x1)
        flow_t = self.get_flow_t(x1, x0, t)
        u_t = self.get_u_t(x1, x0, t)
        return (model(t.squeeze(), flow_t) - u_t).square().mean()

        
        