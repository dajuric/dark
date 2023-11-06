# refs: https://agustinus.kristia.de/techblog/2016/07/18/convnet-maxpool-layer/

import cupy as cp
from .util import im2col, col2im

def maxpool2d_forward(X, f, padding, stride):
    m, n_C, n_H, n_W = X.shape
    
    X_col = im2col(X, f, f, stride, padding)
    X_col = X_col.reshape(n_C, f * f, -1)
    X_col = cp.moveaxis(X_col, 1, 0)
    X_col = X_col.reshape(f * f, -1)
    
    max_idx = cp.argmax(X_col, axis=0, keepdims=True)
    out = X_col[max_idx, range(max_idx.size)]
    
    # Reshape A_pool properly.
    n_H = int((n_H + 2 * padding - f) / stride) + 1
    n_W = int((n_W + 2 * padding - f) / stride) + 1
    
    out = cp.array(cp.hsplit(out, m))
    out = out.reshape(m, n_C, n_H, n_W)
    
    #max_idx = cp.array(cp.hsplit(max_idx, m))
    #max_idx = max_idx.reshape(m, n_C, n_H, n_W)
    
    return out, max_idx

def maxpool2d_backward(dout, X, X_ind, f, padding, stride):
    m, n_C, _, _ = X.shape

    dX = cp.zeros_like(X).reshape(f * f, -1)
    dX[range(X_ind.size), X_ind] = dout.transpose(2, 3, 0, 1).ravel()
    dX = cp.moveaxis(dX, 1, 0)
    dX = dX.reshape(n_C, f * f, -1)
    dX = dX.reshape(n_C * f * f, -1)
    
    dX = col2im(dX, X.shape, f, f, stride, padding)
    return dX
