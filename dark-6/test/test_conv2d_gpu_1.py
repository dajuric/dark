import cupy as cp
from cupyx.scipy.signal import correlate2d
from numba import jit, njit, prange
from ttictoc import tic,toc

def _swap_conv_args_needed(tensor, kernel, padding):
    sb, sc, sh, sw = tensor.shape
    kb, kc, kh, kw = kernel.shape

    oh = sh - kh + 2 * padding + 1
    ow = sw - kw + 2 * padding + 1

    if oh < 0 or ow < 0:
        return kernel, tensor

    return tensor, kernel 

def _pad(image, padding):
    sh, sw = image.shape
    padded = cp.zeros((sh + padding*2, sw + padding*2))
    
    padded[padding: sh + padding, padding: sw + padding] = image
    return padded

def _corr2d_im(image, kernel, padding):
    image = _pad(image, padding)
    image = correlate2d(image, kernel, 'valid')
    return image

def corr2d(tensor, kernel, padding):
    assert len(tensor.shape) == 4 #b, c, w, h
    assert len(kernel.shape) == 4 #b, c, w, h

    tensor, kernel = _swap_conv_args_needed(tensor, kernel, padding)

    sb, sc, sh, sw = tensor.shape
    kb, kc, kh, kw = kernel.shape
    assert sc == kc

    oh = sh - kh + 2 * padding + 1
    ow = sw - kw + 2 * padding + 1
    out = cp.zeros((sb, kb, oh, ow), dtype=cp.float64)

    for bsIdx in range(sb):
        for bkIdx in range(kb):
            for cIdx in range(sc):
                out[bsIdx, bkIdx, :, :] += _corr2d_im(tensor[bsIdx, cIdx, :, :], kernel[bkIdx, cIdx, :, :], padding)

    return out


im = cp.random.random((128, 3, 512, 512))
k  = cp.random.random((1, 3, 5, 5))

corr2d(im, k, 0)

tic()
res = corr2d(im, k, 0)
print(toc())