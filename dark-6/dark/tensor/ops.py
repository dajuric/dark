backends = ["cpu"]
try:
    from cupy import *
    backends.append("cuda")
except:
    from numpy import *
    seterr(over='raise')
    
from .conv_cpu import corr2d_cpu
from .conv_gpu import corr2d_gpu
from .pool_cpu import max_pool_2d_cpu, max_unpool_2d_cpu
from .pool_gpu import max_pool_2d_gpu, max_unpool_2d_gpu
        

def is_cuda():
    return "cuda" in backends

def is_cpu():
    return not is_cuda()

def corr2d(tensor, kernel, padding):
    if is_cpu():
        return corr2d_cpu(tensor, kernel, padding)
    else:
        return corr2d_gpu(tensor, kernel, padding)
    
def conv2d(tensor, kernel, padding):
    kernel = flip(flip(kernel, axis=-2), axis=-1)
    return corr2d(tensor, kernel, padding)
    
def max_pool_2d(tensor, n):
    if is_cpu():
        return max_pool_2d_cpu(tensor, n)
    else:
        return max_pool_2d_gpu(tensor, n)
    
def max_unpool_2d(dldy, x, n):
    if is_cpu():
        return max_unpool_2d_cpu(dldy, x, n)
    else:
        return max_unpool_2d_gpu(dldy, x, n)