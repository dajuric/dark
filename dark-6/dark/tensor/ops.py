_is_cuda = False
try:
    from cupy import *
    from .conv_gpu import corr2d
    from .pool_gpu import max_pool_2d, max_unpool_2d
    _is_cuda = True
except:
    from numpy import *
    from .conv_cpu import corr2d
    from .pool_cpu import max_pool_2d, max_unpool_2d
    seterr(over='raise')
    
def is_cuda():
    return _is_cuda
    
def conv2d(tensor, kernel, padding=0, stride=1):  
    kernel = flip(flip(kernel, axis=-2), axis=-1)
    result = corr2d(tensor, kernel, padding, stride)

    return result