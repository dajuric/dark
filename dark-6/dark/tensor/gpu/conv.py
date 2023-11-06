import cupy as cp
from .util import im2col, col2im

def conv2d_forward(X, W, padding, stride):
    m,  _, xH, xW = X.shape
    nF, _, HF, WF = W.shape 

    X_col = im2col(X, HF, WF, stride, padding)
    w_col = W.reshape((nF, -1))

    # Perform matrix multiplication.
    out = w_col @ X_col
    
    # Reshape back matrix to image.
    n_H = int((xH + 2 * padding - HF) / stride) + 1
    n_W = int((xW + 2 * padding - WF) / stride) + 1
    
    out = cp.array(cp.hsplit(out, m)).reshape((m, nF, n_H, n_W))
    return out

def conv2d_backward(dout, X, W, stride, padding):
    m, _, _, _ = X.shape
    oF, iF, HF, WF = W.shape 
    
    X_col = im2col(X, HF, WF, stride, padding)
    w_col = W.reshape((oF, -1))
   
    # Reshape dout properly.
    dout = dout.reshape(dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
    dout = cp.array(cp.vsplit(dout, m))
    dout = cp.concatenate(dout, axis=-1)
    
    # Perform matrix multiplication between reshaped dout and w_col to get dX_col.
    dX_col = w_col.T @ dout
    
    # Perform matrix multiplication between reshaped dout and X_col to get dW_col.
    dw_col = dout @ X_col.T
    
    # Reshape back to image (col2im).
    dX = col2im(dX_col, X.shape, HF, WF, stride, padding)
    
    # Reshape dw_col into dw.
    dW = dw_col.reshape((dw_col.shape[0], iF, HF, WF))

    return dX, dW
