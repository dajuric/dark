import os
import pickle
from copy import deepcopy
from .. import tensor as dt

def _dt_to_numpy(obj):
    if "cupy.ndarray" in str(obj.__class__):
        return obj.get()
    
    return obj

def _numpy_to_dt(obj):
    if "numpy.ndarray" in str(obj.__class__):
        return dt.asarray(obj)
    
    return obj  

def _convert(obj, convert_fn):
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = _convert(obj[i], convert_fn)
        return obj

    if isinstance(obj, tuple):
        obj = _convert(list(obj), convert_fn)
        return tuple(obj)

    if isinstance(obj, dict):
        for k in obj.keys():
            obj[k] = _convert(obj[k], convert_fn)
        return obj

    if hasattr(obj, "__dict__") == False:
        return convert_fn(obj)

    for key in obj.__dict__:
        obj.__dict__[key] = _convert(obj.__dict__[key], convert_fn)

    return convert_fn(obj)


def load(filename):
    obj = pickle.load(open(filename, "rb"))
    return obj

def save(obj, filename):
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)

    pickle.dump(obj, open(filename, "wb"))