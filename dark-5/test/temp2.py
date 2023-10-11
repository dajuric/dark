import numpy as np
import scipy.signal as sps
from ttictoc import tic,toc

def _swap_conv_args_needed(tensor, kernel, padding):
    sb, sc, sh, sw = tensor.shape
    kb, kc, kh, kw = kernel.shape

    oh = sh - kh + 2 * padding + 1
    ow = sw - kw + 2 * padding + 1

    if oh < 0 or ow < 0:
        return kernel, tensor

    return tensor, kernel 

def _get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0

    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)

    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = _get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def _corr2d(tensor, kernel, padding):
    assert len(tensor.shape) == 4 #b, c, w, h
    assert len(kernel.shape) == 4 #b, c, w, h

    tensor, kernel = _swap_conv_args_needed(tensor, kernel, padding)

    sb, sc, sh, sw = tensor.shape
    kb, kc, kh, kw = kernel.shape
    assert sc == kc

    oh = sh - kh + 2 * padding + 1
    ow = sw - kw + 2 * padding + 1

    # https://agustinus.kristia.de/techblog/2016/07/16/convnet-conv-layer/
    X_col = im2col_indices(tensor, kh, kw, padding=padding)
    W_col = kernel.reshape(kb, -1)

    out = W_col @ X_col
    out = out.reshape(kb, oh, ow, sb)
    out = out.transpose(3, 0, 1, 2)

    return out


im = np.random.random((64, 3, 256, 256))
k  = np.random.random((1, 3, 5, 5))

_corr2d(im, k, 5)

tic()
res = _corr2d(im, k, 5)
print(toc())