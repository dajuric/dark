from dark.nn import module
import numpy as np

def _calc_in_out_dims(t_shape):
    assert len(t_shape) > 1

    in_maps  = t_shape[1]
    out_maps = t_shape[0]
    
    if len(t_shape) > 2:
        s = np.prod(np.array(t_shape[2:]))
        in_maps  *= s
        out_maps *= s

    return in_maps, out_maps

# the idea is taken from the Pytorch code (torch.nn.init.py)
def xavier_uniform_(tensor):
    t_shape = tensor.value.shape
    in_maps, out_maps = _calc_in_out_dims(t_shape)

    std = np.sqrt(2.0 / (in_maps + out_maps))
    a = np.sqrt(3.0) * std
    tensor.value = np.random.uniform(-a, +a, size=t_shape)


def random_uniform_(tensor, a, b):
    t_shape = tensor.value.shape
    tensor.value = (b - a) * np.random.rand(*t_shape) + a


def default_init_weights(m):
    if isinstance(m, module.Conv2d):
        xavier_uniform_(m.weights)
        xavier_uniform_(m.bias)

    if isinstance(m, module.Linear):
        stdv = 1. / np.sqrt(m.weights.value.shape[1])
        random_uniform_(m.weights, -stdv, stdv)
        random_uniform_(m.bias, -stdv, stdv)