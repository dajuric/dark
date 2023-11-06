import numpy as np
from numba import njit, prange

@njit()
def _pad(tensor, padding):
    sb, sc, sh, sw = tensor.shape
    padded = np.zeros(sb, sc, (sh + padding*2, sw + padding*2))
    
    padded[:, :, padding: sh + padding, padding: sw + padding] = tensor
    return padded

#taken and modified: https://stackoverflow.com/questions/2448015/2d-convolution-using-python-and-numpy
@njit()
def _corr2d_im(image, kernel, stride):
    kh, kw = kernel.shape
    sh, sw = image.shape

    oh = (sh - kh) // stride + 1
    ow = (sw - kw) // stride + 1

    res = np.zeros((oh, ow), dtype=np.float64)
    for y in range(0, oh, stride):
        for x in range(0, ow, stride):
            res[y, x] = np.sum(image[y: y + kh, x: x + kw] * kernel)

    return res

@njit(parallel=True)
def conv2d_forward(tensor, kernel, padding, stride):
    assert len(tensor.shape) == 4 #b, c, w, h
    assert len(kernel.shape) == 4 #b, c, w, h

    tb, tc, th, tw = tensor.shape
    kb, kc, kh, kw = kernel.shape
    assert tc == kc

    oh = int((th - kh) / stride + 1)
    ow = int((tw - kw) / stride + 1)
    out = np.zeros((tb, kb, oh, ow), dtype=np.float64)
    tensor = _pad(tensor, padding)

    for bsIdx in prange(tb):
        for bkIdx in range(kb):
            for cIdx in range(tc):
                out[bsIdx, bkIdx, :, :] += _corr2d_im(tensor[bsIdx, cIdx, :, :], kernel[bkIdx, cIdx, :, :], stride)

    return out