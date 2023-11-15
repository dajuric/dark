_is_cuda = False
try:
    from cupy import *
    from .gpu.conv import conv2d_forward, conv2d_backward
    from .gpu.pool import maxpool2d_forward, maxpool2d_backward
    _is_cuda = True
except:
    from numpy import *
    #from .cpu.conv import conv2d_forward #, conv2d_backward
    #from .cpu.pool import maxpool2d_forward, maxpool2d_backward
    seterr(over='raise')
    
def is_cuda():
    return _is_cuda

def sigmoid(x):
    return 1 / (1 + exp(-x))