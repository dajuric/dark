import cupy as cp
from cupy.lib.stride_tricks import as_strided

# https://medium.com/latinxinai/vectorized-convolution-operation-using-numpy-b122fd52fba3
def corr2d(tensor, kernel, padding, stride):
    assert len(tensor.shape) == 4 #b, c, w, h
    assert len(kernel.shape) == 4 #b, c, w, h

    tensor = cp.pad(tensor, [(0, 0), (0, 0), (padding, padding), (padding, padding)])
    
    tb, tc, th, tw = tensor.shape
    kb, kc, kh, kw = kernel.shape
    sb, sc, sh, sw = tensor.strides
    
    oh = int((th - kh) / stride + 1)
    ow = int((tw - kw) / stride + 1)

    out_shape   = (tb, tc, oh, ow, kh, kw)
    out_strides = (sb, sc, stride * sh, stride * sw, sh, sw) 
    windowed_tensor = as_strided(tensor, out_shape, out_strides)

    out = cp.einsum('bchwkt,fckt->bfhw', windowed_tensor, kernel, optimize=True) 
    #out = cp.tensordot(windowed_tensor, kernel, axes=3)
    return out

