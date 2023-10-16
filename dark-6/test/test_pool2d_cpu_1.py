import numpy as np
from numba import njit, prange
from ttictoc import tic,toc

@njit()
def _max_pool_im(im, n):
    h, w = im.shape
    out = np.zeros((h // n, w // n), dtype=np.float64)

    for y in range(0, h-n+1, n):
        for x in range(0, w-n+1, n):
            out[y // n, x // n] = np.max(im[y:y+n, x:x+n])

    return out

@njit(parallel=True)
def max_pool_2d(tensor, n):
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



im = np.random.random((128, 3, 512, 512))

max_pool_2d(im, 2)

tic()
res = max_pool_2d(im, 2)
print(toc())