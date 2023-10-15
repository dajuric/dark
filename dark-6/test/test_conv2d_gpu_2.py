from ttictoc import tic,toc
import cupy as xp
from cupy.lib.stride_tricks import as_strided

# https://stackoverflow.com/questions/65461409/convolutional-layer-in-python-using-numpy-with-strides
# https://stackoverflow.com/questions/56085669/convolutional-layer-in-python-using-numpy
# https://stackoverflow.com/questions/48097941/strided-convolution-of-2d-in-numpy
def corr2d2(a, b):
    a = xp.rollaxis(a, 1, 4)
    b = xp.rollaxis(b, 1, 4); b = xp.rollaxis(b, 0, 4)
    
    Hout = a.shape[1] - b.shape[0] + 1
    Wout = a.shape[2] - b.shape[1] + 1

    a = as_strided(a, (a.shape[0], Hout, Wout, b.shape[0], b.shape[1], a.shape[3]), a.strides[:3] + a.strides[1:])

    return xp.tensordot(a, b, axes=3)


im = xp.random.random((128, 3, 512, 512))
k  = xp.random.random((1, 3, 5, 5))

tic()
res = corr2d2(im, k)
print(toc())