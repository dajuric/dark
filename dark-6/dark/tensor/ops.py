import os
use_cpu = os.environ.get("USE_CPU") in ["1", "True"]
_is_cuda = False

try:
    if use_cpu: raise Exception("CPU forced")
    
    from cupy import *
    from .gpu.conv import conv2d, conv2d_grad
    from .gpu.pool import max_pool2d, max_unpool2d
    _is_cuda = True
except:
    from numpy import *
    from .cpu.conv import conv2d, conv2d_grad
    from .cpu.pool import max_pool2d, max_unpool2d
    seterr(over='raise')
    
def is_cuda():
    return _is_cuda

def numpy(x):
    if x is None:
        return None
    
    if x.__class__.__module__ == "numpy":
        return x
    
    return x.get()

def unsqueeze(x, axis):
    return expand_dims(x, axis)

def sigmoid(x):
    return 1 / (1 + exp(-x))