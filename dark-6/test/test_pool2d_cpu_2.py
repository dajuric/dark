import numpy as np
from numba import njit, prange
from ttictoc import tic,toc


def max_pool_2d(tensor, n):
    assert len(tensor.shape) == 4 #b, c, w, h

    sb, sc, sh, sw = tensor.shape
    assert sh % n == 0 and sw % n == 0

    oh = sh // n
    ow = sw // n
    
    #tensor = tensor.reshape(sb, sc, oh, n, ow, n).swapaxes(3, 4).reshape(sb, sc, oh, ow, n * n)
    #out = np.amax(tensor, axis=4)
   
    out = tensor.reshape(sb, sc, oh, n, ow, n).max(axis=(3, 5))
    return out



im = np.random.random((128, 3, 512, 512))

max_pool_2d(im, 2)

tic()
res = max_pool_2d(im, 2)
print(toc())