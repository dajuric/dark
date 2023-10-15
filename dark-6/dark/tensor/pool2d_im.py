from .ops import *
from numba import jit, cuda

#@jit(nopython=is_cpu(), forceobj=not is_cpu())
cuda.jit()
def max_pool2d_im(im, n):
    h, w = im.shape
    out = cp.zeros((h // n, w // n), dtype=cp.float64)

    for y in range(0, h-n+1, n):
        for x in range(0, w-n+1, n):
            out[y // n, x // n] = cp.max(im[y:y+n, x:x+n])

    return out


#@jit(nopython=is_cpu(), forceobj=not is_cpu())
cuda.jit()
def _unravel_index(idx, shape):
    h, w = shape
    r = idx // w
    c = idx % w

    return r, c

#@jit(nopython=is_cpu(), forceobj=not is_cpu())
cuda.jit()
def max_unpool2d_im(dldy, im, n):
    h, w = im.shape
    out = cp.zeros(im.shape, dtype=cp.float64)

    for y in range(0, h-n+1, n):
        for x in range(0, w-n+1, n):
            maxIdx = cp.argmax(im[y:y+n, x:x+n])
            maxIdx = _unravel_index(maxIdx, (n, n))
            out[y:y+n, x:x+n][maxIdx] = dldy[y // n, x // n]

    return out

