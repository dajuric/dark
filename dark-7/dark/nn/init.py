import dark.tensor as dt
import dark.nn as nn
from dark import *

def _calc_in_out_dims(t_shape):
    assert len(t_shape) > 1

    in_maps  = t_shape[1]
    out_maps = t_shape[0]
    
    if len(t_shape) > 2:
        s = dt.prod(dt.array(t_shape[2:]))
        in_maps  *= s
        out_maps *= s

    return in_maps, out_maps

# the idea is taken from the Pytorch code (torch.nn.init.py)
def xavier_uniform_(tensor):
    t_shape = tensor.data.shape
    in_maps, out_maps = _calc_in_out_dims(t_shape)

    std = dt.sqrt(2.0 / (in_maps + out_maps))
    a = dt.sqrt(3.0) * std
    tensor.data = dt.random.uniform(-a, +a, size=t_shape)


def random_uniform_(tensor, a, b):
    t_shape = tensor.data.shape
    tensor.data = (b - a) * dt.random.rand(*t_shape) + a


def default_init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        xavier_uniform_(m.weights)
        xavier_uniform_(m.bias)

    if isinstance(m, nn.Linear):
        stdv = 1. / dt.sqrt(m.weights.data.shape[1])
        random_uniform_(m.weights, -stdv, stdv)
        random_uniform_(m.bias, -stdv, stdv)