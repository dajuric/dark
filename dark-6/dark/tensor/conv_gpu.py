import cupy as cp

def get_indices(X_shape, HF, WF, stride, pad):
    # get input size
    m, n_C, n_H, n_W = X_shape

    # get output size
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1
  
    # ----Compute matrix of index i----

    # Level 1 vector.
    level1 = cp.repeat(cp.arange(HF), WF)
    # Duplicate for the other channels.
    level1 = cp.tile(level1, n_C)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride * cp.repeat(cp.arange(out_h), out_w)
    # Create matrix of index i at every levels for each channel.
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # ----Compute matrix of index j----
    
    # Slide 1 vector.
    slide1 = cp.tile(cp.arange(WF), HF)
    # Duplicate for the other channels.
    slide1 = cp.tile(slide1, n_C)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride * cp.tile(cp.arange(out_w), out_h)
    # Create matrix of index j at every slides for each channel.
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    # ----Compute matrix of index d----

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = cp.repeat(cp.arange(n_C), HF * WF).reshape(-1, 1)

    return i, j, d

def im2col(X, HF, WF, stride, pad):
    # Padding
    X_padded = cp.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    
    # Multi-dimensional arrays indexing.
    cols = X_padded[:, d, i, j]
    cols = cp.concatenate(cols, axis=-1)
    return cols

def col2im(dX_col, X_shape, HF, WF, stride, pad):
    # Get input size
    N, D, H, W = X_shape
    # Add padding if needed.
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = cp.zeros((N, D, H_padded, W_padded))
    
    # Index matrices, necessary to transform our input image into a matrix. 
    i, j, d = get_indices(X_shape, HF, WF, stride, pad)
    
    # Retrieve batch dimension by spliting dX_col N times: (X, Y) => (N, X, Y)
    dX_col_reshaped = cp.array(cp.hsplit(dX_col, N))
    
    # Reshape our matrix back to image.
    # slice(None) is used to produce the [::] effect which means "for every elements".
    cp.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
    
    # Remove padding from new image if needed.
    if pad == 0:
        return X_padded
    elif type(pad) is int:
        return X_padded[:, :, pad:-pad, pad:-pad]


def conv_forward(X, W, padding, stride):
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

def conv_backward(dout, X, W, stride, padding):
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


def maxpool_forward(X, f, padding, stride):
    m, n_C_prev, n_H_prev, n_W_prev = X.shape
    n_C = n_C_prev
    
    n_H = int((n_H_prev + 2 * padding - f)/ stride) + 1
    n_W = int((n_W_prev + 2 * padding - f)/ stride) + 1

    X_col = im2col(X, f, f, stride, padding)
    X_col = X_col.reshape(n_C, X_col.shape[0]//n_C, -1)
    A_pool = cp.max(X_col, axis=1)
    
    # Reshape A_pool properly.
    A_pool = cp.array(cp.hsplit(A_pool, m))
    A_pool = A_pool.reshape(m, n_C, n_H, n_W)

    return A_pool

def maxpool_backward(dout, X, f, padding, stride):
    m, n_C_prev, n_H_prev, n_W_prev = X.shape
    n_C = n_C_prev

    dout_flatten = dout.reshape(n_C, -1) / (f * f)
    dX_col = cp.repeat(dout_flatten, f * f, axis=0)
    dX = col2im(dX_col, X.shape, f, f, stride, padding)
    
    # Reshape dX properly.
    dX = dX.reshape(m, -1)
    dX = cp.array(cp.hsplit(dX, n_C_prev))
    dX = dX.reshape(m, n_C_prev, n_H_prev, n_W_prev)
    return dX


# x = cp.random.randn(*(5, 4, 128, 128))
# k = cp.random.randn(*(2, 4, 3, 3))
# out = conv_forward(x, k, 1, 1)
# dx, dk = conv_backward(out, x, k, 1, 1)
# print()