import cupy as cp
#import numpy as cp
from numba import njit, prange
from ttictoc import tic,toc

# https://tinynet.autoai.org/_/downloads/en/latest/pdf/
# https://github.com/mratsim/Arraymancer/issues/174
# https://stackoverflow.com/questions/31972990/increasing-speed-of-a-pure-numpy-scipy-convolutional-neural-network-implementati
def max_pool_2d(tensor, n):
    assert len(tensor.shape) == 4 #b, c, w, h

    sb, sc, sh, sw = tensor.shape
    assert sh % n == 0 and sw % n == 0

    oh = sh // n
    ow = sw // n
    
    #tensor = tensor.reshape(sb, sc, oh, n, ow, n).swapaxes(3, 4).reshape(sb, sc, oh, ow, n * n)
    #out = np.amax(tensor, axis=4)
   
    out = tensor.reshape(sb, sc, oh, n, ow, n).max(axis=(3, 5))
    return out

# https://numbersmithy.com/2d-and-3d-pooling-using-numpy/#The_unpooling_operation
def max_unpool_2d(dldy, tensor, n):
    assert len(dldy.shape) == 4 #b, c, w, h
    assert len(tensor.shape) == 4 #b, c, w, h

    sb, sc, sh, sw = tensor.shape
    assert sh % n == 0 and sw % n == 0

    oh = sh // n
    ow = sw // n

    tensor = tensor.reshape(sb, sc, oh, n, ow, n).swapaxes(3, 4).reshape(sb, sc, oh, ow, n * n)
    indices = tensor.argmax(axis=4)

    out = cp.zeros_like(tensor)
    out.reshape(-1, n*n)[cp.arange(sb * sc * oh * ow), indices.reshape(-1)] = dldy.reshape(-1)
    out = out.reshape(sb, sc, oh, ow, n * n).reshape(sb, sc, oh, ow, n, n).swapaxes(3, 4).reshape(sb, sc, sh, sw)

    # out = cp.zeros_like(tensor)
    # out[:, :, cp.arange(oh), cp.arange(ow), indices.reshape(-1)] = dldy
    # out = out.reshape(sb, sc, oh, ow, n * n).reshape(sb, sc, oh, ow, n, n).swapaxes(3, 4).reshape(sb, sc, sh, sw)

    return out




im = cp.random.random((128, 3, 512, 512))
# im = cp.array([[1, 0, 2, 0],
#                [5, 0, 3, 6]])
# im = cp.expand_dims(im, 0)
# im = cp.expand_dims(im, 0)

dldy = max_pool_2d(im, 2)
max_unpool_2d(dldy, im, 2)

tic()
#res = max_pool_2d(im, 2)
max_unpool_2d(dldy, im, 2)
print(toc())