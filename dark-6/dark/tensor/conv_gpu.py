import cupy as cp
from cupy.lib.stride_tricks import as_strided

def _swap_conv_args_needed(tensor, kernel, padding):
    sb, sc, sh, sw = tensor.shape
    kb, kc, kh, kw = kernel.shape

    oh = sh - kh + 2 * padding + 1
    ow = sw - kw + 2 * padding + 1

    if oh < 0 or ow < 0:
        return kernel, tensor

    return tensor, kernel 

def corr2d_gpu(tensor, kernel, padding):
    assert len(tensor.shape) == 4 #b, c, w, h
    assert len(kernel.shape) == 4 #b, c, w, h

    tensor, kernel = _swap_conv_args_needed(tensor, kernel, padding)
    tensor = cp.pad(tensor, [(0, 0), (0, 0), (padding, padding), (padding, padding)])
    
    tensor = cp.rollaxis(tensor, 1, 4)
    kernel = cp.rollaxis(kernel, 1, 4); kernel = cp.rollaxis(kernel, 0, 4)
    
    tb, th, tw, tc = tensor.shape
    kh, kw, kc, kb = kernel.shape
    sb, sh, sw, sc = tensor.strides
    
    oh = th - kh + 1
    ow = tw - kw + 1

    out_shape = (tb, oh, ow, kh, kw, tc)
    strides   = (sb, sh, sw,  sh, sw, sc) 
    tensor = as_strided(tensor, out_shape, strides)
    
    out = cp.tensordot(tensor, kernel, axes=3)
    out = cp.rollaxis(out, 3, 1)
    return out

