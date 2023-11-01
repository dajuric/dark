# refs: https://agustinus.kristia.de/techblog/2016/07/18/convnet-maxpool-layer/

import cupy as cp

def max_pool_2d(tensor, n):
    assert len(tensor.shape) == 4 #b, c, w, h

    sb, sc, sh, sw = tensor.shape
    assert sh % n == 0 and sw % n == 0

    oh = sh // n
    ow = sw // n
    
    out = tensor.reshape(sb, sc, oh, n, ow, n).max(axis=(3, 5))
    return out

def max_unpool_2d(dldy, tensor, n):
    assert len(dldy.shape) == 4 #b, c, w, h
    assert len(tensor.shape) == 4 #b, c, w, h

    sb, sc, sh, sw = tensor.shape
    assert sh % n == 0 and sw % n == 0

    oh = sh // n
    ow = sw // n

    tensor = tensor.reshape(sb, sc, oh, n, ow, n).swapaxes(3, 4).reshape(sb, sc, oh, ow, n * n)
    indices = tensor.argmax(axis=4).reshape(-1)

    out = cp.zeros_like(tensor).reshape(-1, n*n)
    out[cp.arange(sb * sc * oh * ow), indices] = dldy.reshape(-1)
    out = out.reshape(sb, sc, oh, ow, n * n).reshape(sb, sc, oh, ow, n, n).swapaxes(3, 4).reshape(sb, sc, sh, sw)

    return out