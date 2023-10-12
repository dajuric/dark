# refs: https://agustinus.kristia.de/techblog/2016/07/18/convnet-maxpool-layer/

from .autodiff import Operation
from .utils import *
import numpy as np
from numba import njit, prange

@njit()
def _max_pool_im(im, n):
    h, w = im.shape
    out = np.zeros((h // n, w // n), dtype=np.float64)

    for y in range(0, h-n+1, n):
        for x in range(0, w-n+1, n):
            out[y // n, x // n] = np.max(im[y:y+n, x:x+n])

    return out

@njit(parallel=True)
def _max_pool_2d(tensor, n):
    assert len(tensor.shape) == 4 #b, c, w, h

    sb, sc, sh, sw = tensor.shape
    assert sh % n == 0 and sw % n == 0

    oh = sh // n
    ow = sw // n
    out = np.zeros((sb, sc, oh, ow), dtype=np.float64)

    for bIdx in prange(sb):
        for cIdx in range(sc):
           out[bIdx, cIdx, :, :] += _max_pool_im(tensor[bIdx, cIdx, :, :], n)

    return out



@njit()
def _unravel_index(idx, shape):
    h, w = shape
    r = idx // w
    c = idx % w

    return r, c

@njit()
def _max_unpool_im(dldy, im, n):
    h, w = im.shape
    out = np.zeros(im.shape, dtype=np.float64)

    for y in range(0, h-n+1, n):
        for x in range(0, w-n+1, n):
            maxIdx = np.argmax(im[y:y+n, x:x+n])
            maxIdx = _unravel_index(maxIdx, (n, n))
            out[y:y+n, x:x+n][maxIdx] = dldy[y // n, x // n]

    return out

@njit(parallel=True)
def _max_unpool_2d(dldy, tensor, n):
    assert len(dldy.shape) == 4 #b, c, w, h
    assert len(tensor.shape) == 4 #b, c, w, h

    sb, sc, sh, sw = tensor.shape
    assert sh % n == 0 and sw % n == 0

    out = np.zeros(tensor.shape, dtype=np.float64)

    for bIdx in prange(sb):
        for cIdx in range(sc):
           out[bIdx, cIdx, :, :] += _max_unpool_im(dldy[bIdx, cIdx, :, :], tensor[bIdx, cIdx, :, :], n)

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

