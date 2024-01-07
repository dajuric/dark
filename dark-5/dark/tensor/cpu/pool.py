# https://agustinus.kristia.de/techblog/2016/07/18/convnet-maxpool-layer/
# https://hackmd.io/@machine-learning/blog-post-cnnumpy-slow

import numpy as np
from numba import njit, prange

@njit()
def _max_pool2d_im(X, size, stride, out, out_idx):
    oh, ow = out.shape
    
    for h in range(oh):
        h_start = h * size
        h_end = h_start + stride

        for w in range(ow):
            w_start = w * size
            w_end = w_start + stride
            patch = X[h_start:h_end, w_start:w_end]

            out[h, w] = np.max(patch)
            out_idx[h, w] = np.argmax(patch)

@njit(parallel=True)
def max_pool2d(X, size, stride): 
    b, c, h, w = X.shape
    h_out = (h - size) / stride + 1
    w_out = (w - size) / stride + 1

    if h_out != int(h_out) or w_out != int(w_out):
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)   
    out = np.zeros((b, c, h_out, w_out))
    out_idx = np.zeros((b, c, h_out, w_out), dtype=np.int32)

    for bIdx in prange(b):
        for cIdx in range(c):
           _max_pool2d_im(X[bIdx, cIdx, ...], size, stride, 
                          out[bIdx, cIdx, ...], out_idx[bIdx, cIdx, ...])

    return out, out_idx



@njit()
def _unravel_index(idx, shape):
    h, w = shape
    r = idx // w
    c = idx % w

    return r, c

@njit()
def _max_unpool2d_im(dout, dout_ind, size, stride, dX):
    oh, ow = dout.shape
   
    for h in range(oh): # Slide the filter vertically.
        h_start = h * size
        h_end = h_start + stride

        for w in range(ow): # Slide the filter horizontally.
            w_start = w * size
            w_end = w_start + stride

            max_idx = _unravel_index(dout_ind[h, w], (size, size))
            dX[h_start:h_end, w_start:w_end][max_idx] = dout[h, w]

@njit(parallel=True)
def max_unpool2d(dout, dout_ind, X_shape, size, stride):
    ob, oc, _, _ = dout.shape
    dX = np.zeros(X_shape)

    for b in prange(ob):
        for c in range(oc):
            _max_unpool2d_im(dout[b, c, ...], dout_ind[b, c, ...], size, stride,
                             dX[b, c, ...])
           
    return dX