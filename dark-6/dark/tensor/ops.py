backends = ["cpu"]

import numpy as np

try:
    import cupy as xp
    from cupyx.scipy.signal import correlate2d
    backends.append("cuda")
except:
    import numpy as xp
        
np.seterr(over='raise')
        
def is_cuda():
    return "cuda" in backends

def is_cpu():
    return not is_cuda()