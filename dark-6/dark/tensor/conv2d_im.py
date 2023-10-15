from .ops import *
from numba import jit, cuda

#@jit(nopython=is_cpu())
@cuda.jit()
def _pad(image, padding):
    sh, sw = image.shape
    padded = cp.zeros((sh + padding*2, sw + padding*2))
    
    padded[padding: sh + padding, padding: sw + padding] = image
    return padded

#taken and modified: https://stackoverflow.com/questions/2448015/2d-convolution-using-python-and-numpy
@jit(nopython=True)
def _corr2d_im_cpu(image, kernel, padding):
    image = _pad(image, padding)

    kh, kw = kernel.shape
    sh, sw = image.shape

    oh = (sh - kh) + 1
    ow = (sw - kw) + 1

    res = cp.zeros((oh, ow), dtype=cp.float64)
    for y in range(0, oh):
        for x in range(0, ow):
            res[y, x] = cp.sum(image[y: y + kh, x: x + kw] * kernel)

    return res

def _corr2d_im_cuda(image, kernel, padding):
    image = _pad(image, padding)
    image = correlate2d(image, kernel, 'valid')
    return image

def corr2d_im(image, kernel, padding):
    if is_cpu():
        return _corr2d_im_cpu(image, kernel, padding)
    else:
        return _corr2d_im_cuda(image, kernel, padding)