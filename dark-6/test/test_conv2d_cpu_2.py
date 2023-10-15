import numpy as xp
from numpy.lib.stride_tricks import as_strided
from ttictoc import tic,toc


# def _swap_conv_args_needed(tensor, kernel, padding):
#     sb, sc, sh, sw = tensor.shape
#     kb, kc, kh, kw = kernel.shape

#     oh = sh - kh + 2 * padding + 1
#     ow = sw - kw + 2 * padding + 1

#     if oh < 0 or ow < 0:
#         return kernel, tensor

#     return tensor, kernel 

# # https://stackoverflow.com/questions/48097941/strided-convolution-of-2d-in-numpy
# # https://github.com/scikit-image/scikit-image/blob/main/skimage/util/shape.py#L97-L247
# def view_as_windows(arr_in, window_shape, step = 1):    
#     step = (step,) * arr_in.ndim

#     slices = tuple(slice(None, None, st) for st in step)
#     window_strides = arr_in.strides

#     indexing_strides = arr_in[slices].strides

#     win_indices_shape = (((xp.array(arr_in.shape) - xp.array(window_shape))
#                           // xp.array(step)) + 1)
#     win_indices_shape = [x.item() for x in win_indices_shape]

#     new_shape = tuple(list(win_indices_shape) + list(window_shape))
#     strides = tuple(list(indexing_strides) + list(window_strides))

#     arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
#     return arr_out

# def corr2d(tensor, kernel, padding=0, step=1):
#     assert len(tensor.shape) == 4 #b, c, w, h
#     assert len(kernel.shape) == 4 #b, c, w, h

#     tensor, kernel = _swap_conv_args_needed(tensor, kernel, padding)
#     tensor4D = view_as_windows(tensor, kernel.shape, step=step)
    
#     result = xp.tensordot(tensor4D, kernel, axes=((6,7),(2,3)))
#     return result
    
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