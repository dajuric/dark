import dark.tensor as dt
import dark.nn as nn
from dark import *

def random_uniform_(tensor, a, b):
    t_shape = tensor.data.shape
    tensor.data = (b - a) * dt.random.rand(*t_shape) + a


def default_init_weights(m):
    if isinstance(m, nn.Linear):
        stdv = 1. / dt.sqrt(m.weights.data.shape[1])
        random_uniform_(m.weights, -stdv, stdv)
        random_uniform_(m.bias, -stdv, stdv)