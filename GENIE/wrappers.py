import torch.nn as nn

from utils.util import add_dimensions



def model_fn(model, x_t, t, y=None, context=None):
    if y is not None:
        if context is not None:
            return model(x_t, t, y=y, context=context)
        else:
            return model(x_t, t, y=y)
    else:
        if context is not None:
            return model(x_t, t, context=context)
        else:
            return model(x_t, t)


class EpsPredictor(nn.Module):
    def __init__(self, model, M, beta_min, beta_d):
        super().__init__()

        self.model = model
        self.M = M
        self.beta_min = beta_min
        self.beta_d = beta_d

    def forward(self, x_t, t, y=None, context=None, return_embeddings=False):
        if return_embeddings:
            return model_fn(self.model, x_t, self.M * t, y, context)
        else:
            return model_fn(self.model, x_t, self.M * t, y, context)[0]

    def eps(self, x_t, t, y=None, context=None, return_embeddings=False):
        return self.forward(x_t, t, y, context, return_embeddings)


class VPredictor(nn.Module):
    def __init__(self, model, M, beta_min, beta_d):
        super().__init__()

        self.model = model
        self.M = M
        self.beta_min = beta_min
        self.beta_d = beta_d

    def forward(self, x_t, t, y=None, context=None, return_embeddings=False):
        if return_embeddings:
            return model_fn(self.model, x_t, self.M * t, y, context)
        else:
            return model_fn(self.model, x_t, self.M * t, y, context)[0]

    def eps(self, x_t, t, y=None, context=None, return_embeddings=False):
        alpha_t = (-.5 * (self.beta_min * t + .5 * self.beta_d * t ** 2.)).exp()
        alpha_t = add_dimensions(alpha_t, len(x_t.shape) - 1)

        model_out = self.forward(x_t, t, y, context, return_embeddings)
        if return_embeddings:
            return alpha_t * model_out[0] + (1. - alpha_t ** 2.).sqrt() * x_t, model_out[1], model_out[2]
        else:
            return alpha_t * model_out + (1. - alpha_t ** 2.).sqrt() * x_t

    