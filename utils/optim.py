import logging
from torch.optim import Adam, Adamax, AdamW, SGD


def get_optimizer(optimizer, params, **kwargs):
    if optimizer == 'Adam':
        if 'beta1' in kwargs and 'beta2' in kwargs:
            optimizer = Adam(params,
                            lr=kwargs['learning_rate'],
                            weight_decay=kwargs['weight_decay'],
                            betas=(kwargs['beta1'], kwargs['beta2']))
        else:
            optimizer = Adam(params,
                            lr=kwargs['learning_rate'],
                            weight_decay=kwargs['weight_decay'])
    elif optimizer == 'FusedAdam':
        try:
            from apex.optimizers import FusedAdam
            optimizer = Adam(params,
                         lr=kwargs['learning_rate'],
                         weight_decay=kwargs['weight_decay'])
        except ImportError:
            logging.info('Apex is not available. Falling back to PyTorch\'s native Adam.')
            return get_optimizer(optimizer, params, kwargs)
    elif optimizer == 'Adamax':
        optimizer = Adamax(params,
                           lr=kwargs['learning_rate'],
                           weight_decay=kwargs['weight_decay'])
    elif optimizer == 'AdamW':
        optimizer = AdamW(params,
                          lr=kwargs['learning_rate'],
                          weight_decay=kwargs['weight_decay'])
    elif optimizer == 'SGD':
        optimizer = SGD(params,
                        lr=kwargs['learning_rate'],
                        momentum=kwargs['momentum'],
                        weight_decay=kwargs['weight_decay'])
    else:
        raise NotImplementedError('Optimizer %s is not supported.' % optimizer)

    return optimizer
