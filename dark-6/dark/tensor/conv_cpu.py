import numpy as np
from numba import njit, prange

@njit()
def _swap_conv_args_needed(tensor, kernel, padding):
    sb, sc, sh, sw = tensor.shape
    kb, kc, kh, kw = kernel.shape

    oh = sh - kh + 2 * padding + 1
    ow = sw - kw + 2 * padding + 1

    if oh < 0 or ow < 0:
        return kernel, tensor

    return tensor, kernel 

@njit()
def _pad(image, padding):
    sh, sw = image.shape
    padded = np.zeros((sh + padding*2, sw + padding*2))
    
    padded[padding: sh + padding, padding: sw + padding] = image
    return padded

#taken and modified: https://stackoverflow.com/questions/2448015/2d-convolution-using-python-and-numpy
@njit()
def _corr2d_im(image, kernel, padding):
    image = _pad(image, padding)

    kh, kw = kernel.shape
    sh, sw = image.shape

    oh = (sh - kh) + 1
    ow = (sw - kw) + 1

    res = np.zeros((oh, ow), dtype=np.float64)
    for y in range(0, oh):
        for x in range(0, ow):
            res[y, x] = np.sum(image[y: y + kh, x: x + kw] * kernel)

    return res

@njit(parallel=True)
def corr2d_cpu(tensor, kernel, padding):
    assert len(tensor.shape) == 4 #b, c, w, h
    assert len(kernel.shape) == 4 #b, c, w, h

    tensor, kernel = _swap_conv_args_needed(tensor, kernel, padding)

    sb, sc, sh, sw = tensor.shape
    kb, kc, kh, kw = kernel.shape
    assert sc == kc

    oh = sh - kh + 2 * padding + 1
    ow = sw - kw + 2 * padding + 1
    out = np.zeros((sb, kb, oh, ow), dtype=np.float64)

    for bsIdx in prange(sb):
        for bkIdx in range(kb):
            for cIdx in range(sc):
                out[bsIdx, bkIdx, :, :] += _corr2d_im(tensor[bsIdx, cIdx, :, :], kernel[bkIdx, cIdx, :, :], padding)

    return out