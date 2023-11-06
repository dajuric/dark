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