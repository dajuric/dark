# refs: https://agustinus.kristia.de/techblog/2016/07/18/convnet-maxpool-layer/

import cupy as cp
from .util import im2col, col2im

def maxpool2d_forward(X, size, stride):
    n, d, h, w = X.shape
    h_out = (h - size) / stride + 1
    w_out = (w - size) / stride + 1

    if not w_out.is_integer() or not h_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_reshaped = X.reshape(n * d, 1, h, w)
    X_col = im2col(X_reshaped, size, size, padding=0, stride=stride)

    max_idx = cp.argmax(X_col, axis=0)
    out = X_col[max_idx, range(max_idx.size)]

    out = out.reshape(h_out, w_out, n, d)
    out = out.transpose(2, 3, 0, 1)

    return out, max_idx

def maxpool2d_backward(dout, X, X_ind, size, stride):
    n, d, w, h = X.shape

    dX_col = cp.zeros_like(X).reshape(size * size, -1)
    dout_col = dout.transpose(2, 3, 0, 1).ravel()

    dX_col[X_ind, range(dout_col.size)] = dout_col
    dX = col2im(dX_col, (n * d, 1, h, w), size, size, padding=0, stride=stride)
    dX = dX.reshape(X.shape)

    return dX
