from .autodiff import Operation
from .utils import *
#from dark.tensor import *
import cupy as cp
from numba import jit, prange, cuda

@cuda.jit()
def _pad(image, padding):
    sh, sw = image.shape
    padded = cp.zeros((sh + padding*2, sw + padding*2))
    
    padded[padding: sh + padding, padding: sw + padding] = image
    return padded

@cuda.jit()
def _corr2d_im_cuda(image, kernel, padding):
    image = _pad(image, padding)
    image = correlate2d(image, kernel, 'valid')
    return image


#@jit(nopython=is_cpu())
@cuda.jit()
def _swap_conv_args_needed(tensor, kernel, padding):
    sb, sc, sh, sw = tensor.shape
    kb, kc, kh, kw = kernel.shape

    oh = sh - kh + 2 * padding + 1
    ow = sw - kw + 2 * padding + 1

    if oh < 0 or ow < 0:
        return kernel, tensor

    return tensor, kernel 

#@jit(nopython=is_cpu(), parallel=is_cpu())
@cuda.jit()
def _corr2d(tensor, kernel, padding):
    assert len(tensor.shape) == 4 #b, c, w, h
    assert len(kernel.shape) == 4 #b, c, w, h

    tensor, kernel = _swap_conv_args_needed(tensor, kernel, padding)

    sb, sc, sh, sw = tensor.shape
    kb, kc, kh, kw = kernel.shape
    assert sc == kc

    oh = sh - kh + 2 * padding + 1
    ow = sw - kw + 2 * padding + 1
    out = cp.zeros((sb, kb, oh, ow))

    for bsIdx in prange(sb):
        for bkIdx in range(kb):
            for cIdx in range(sc):
                out[bsIdx, bkIdx, :, :] += _corr2d_im_cuda(tensor[bsIdx, cIdx, :, :], kernel[bkIdx, cIdx, :, :], padding)

    return out

def _conv2d(tensor, kernel, mode):
    kernel = cp.flip(cp.flip(kernel, axis=-2), axis=-1) #flip horizonal and vertical
    return _corr2d(tensor, kernel, mode)


class Conv2D(Operation):

    @staticmethod
    def _f(s, k, **kwargs):
        return _corr2d[32, 32](s, k, kwargs["padding"])

    @staticmethod
    def _df(dldy, y, s, k):
        p = Conv2D._get_padding(s.shape[-1], dldy.shape[-1], k.shape[-1])
        dlds = _conv2d(dldy, k.transpose((1, 0, 2, 3)), abs(p))

        p = Conv2D._get_padding(k.shape[-1], dldy.shape[-1], s.shape[-1])
        dldk = _conv2d(dldy.transpose((1, 0, 2, 3)), s.transpose((1, 0, 2, 3)), abs(p))
        return dlds, dldk

    @staticmethod
    def _get_padding(o, s, k):
        p = (o - s + k - 1) // 2
        return p

def conv2d(s, k, padding = 0):
    return Conv2D.apply(s, k, padding = padding)
