# refs: https://agustinus.kristia.de/techblog/2016/07/18/convnet-maxpool-layer/

from .autodiff import Operation
from .utils import *
from dark.tensor import *
from numba import jit, prange, cuda

#@jit(nopython=is_cpu(), parallel=is_cpu())
cuda.jit()
def _max_pool_2d(tensor, n):
    assert len(tensor.shape) == 4 #b, c, w, h

    sb, sc, sh, sw = tensor.shape
    assert sh % n == 0 and sw % n == 0

    oh = sh // n
    ow = sw // n
    out = cp.zeros((sb, sc, oh, ow), dtype=cp.float64)

    for bIdx in prange(sb):
        for cIdx in range(sc):
           out[bIdx, cIdx, :, :] += max_pool2d_im(tensor[bIdx, cIdx, :, :], n)

    return out

#@jit(nopython=is_cpu(), parallel=is_cpu())
cuda.jit()
def _max_unpool_2d(dldy, tensor, n):
    assert len(dldy.shape) == 4 #b, c, w, h
    assert len(tensor.shape) == 4 #b, c, w, h

    sb, sc, sh, sw = tensor.shape
    assert sh % n == 0 and sw % n == 0

    out = cp.zeros(tensor.shape, dtype=cp.float64)

    for bIdx in prange(sb):
        for cIdx in range(sc):
           out[bIdx, cIdx, :, :] += max_unpool2d_im(dldy[bIdx, cIdx, :, :], tensor[bIdx, cIdx, :, :], n)

    return out

class MaxPool2D(Operation):

    @staticmethod
    def _f(x, **kwargs):
        return _max_pool_2d(x, kwargs["kernel_size"])

    @staticmethod
    def _df(dldy, y, x):
        n = x.shape[-1] // y.shape[-1]
        dldx = _max_unpool_2d(dldy, x, n)
        return [dldx]

def max_pool2d(x, kernel_size = 2):
    return MaxPool2D.apply(x, kernel_size = kernel_size)

