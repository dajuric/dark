from dark.nn import module
import numpy as np

def random_uniform_(tensor, a, b):
    t_shape = tensor.value.shape
    tensor.value = (b - a) * np.random.rand(*t_shape) + a


def default_init_weights(m):
    if isinstance(m, module.Linear):
        stdv = 1. / np.sqrt(m.weights.value.shape[1])
        random_uniform_(m.weights, -stdv, stdv)
        random_uniform_(m.bias, -stdv, stdv)