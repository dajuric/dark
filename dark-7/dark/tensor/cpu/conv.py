import numpy as np
from numba import njit, prange

@njit()
def _pad(tensor, padding):
    sb, sc, sh, sw = tensor.shape
    padded = np.zeros((sb, sc, sh + padding*2, sw + padding*2))
    
    padded[:, :, padding: sh + padding, padding: sw + padding] = tensor
    return padded


@njit()
def _conv2d_im(X, W, stride, out):
    _, kh, kw = W.shape
    oh, ow = out.shape
    
    for h in range(oh): # Slide the filter vertically.
        h_start = h * stride
        h_end = h_start + kh

        for w in range(ow): # Slide the filter horizontally.                
            w_start = w * stride
            w_end = w_start + kw

            # Element wise multiplication + sum.
            out[h, w] = np.sum(X[:, h_start:h_end, w_start:w_end] * W)

@njit(parallel=True)
def conv2d(X, W, stride, padding):
    tb, tc, th, tw = X.shape
    kb, kc, kh, kw = W.shape
    assert tc == kc

    # Define output size.
    oh = int((th + 2 * padding - kh) / stride) + 1
    ow = int((tw + 2 * padding - kw) / stride) + 1

    out = np.zeros((tb, kb, oh, ow))
    X = _pad(X, padding)

    for b in prange(tb): # For each image.
        for oc in range(kb): # For each output channel.
            _conv2d_im(X[b, ...], W[oc, ...], stride, out[b, oc, ...])
           
    return out 


@njit()
def _conv2d_grad_im(dout, X, W, stride, dX, dW):
    oh, ow = dout.shape
    _, kh, kw = W.shape
    
    for h in range(oh): # Slide the filter vertically.
        h_start = h * stride
        h_end = h_start + kh

        for w in range(ow): #  Slide the filter horizontally.
            w_start = w * stride
            w_end = w_start + kw

            dW[:, :, :] += dout[h, w] * X[:, h_start:h_end, w_start:w_end]
            dX[:, h_start:h_end, w_start:w_end] += dout[h, w] * W
                        
@njit(parallel=True)
def conv2d_grad(dout, X, W, stride, padding):
    tb, tc, th, tw = X.shape
    ob, oc, oh, ow = dout.shape

    X = _pad(X, padding)
    dX = np.zeros(X.shape)
    dWt = np.zeros((tb, ) + W.shape, dtype=np.float64) #thread-safe dW
    
    for i in prange(tb): # For each image.  
        for c in range(oc): # For each channel.
           _conv2d_grad_im(dout[i, c, ...], X[i, ...], W[c, ...], stride, 
                           dX[i, ...], dWt[i, c, ...])

    dX = dX[:, :, padding:th+padding, padding:tw+padding]
    dW = dWt.sum(axis=0)
    return dX, dW
            