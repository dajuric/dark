#import os
#os.environ["USE_CPU"] = "True"

import dark
import dark.tensor as dt

def approx_grad(X, function, eps=1e-6):
    grad = dt.zeros_like(X)
    
    for i in range(len(X)):
        X[i] += eps
        fn_2 = function(X)
        
        X[i] -= 2 * eps
        fn_1 = function(X)
        
        grad[i] = (fn_2 - fn_1) / (2 * eps)
        X[i] += eps
        
    return grad

def func(x, k, stride, padding):
    out = dark.conv_transpose2d(x, k, stride, padding)
    out = dark.pow(out, 2)
    return out

def check(X_shape, k_shape, stride, padding):
    x = dark.Parameter(dt.random.random(X_shape) * 2 - 1)
    k = dark.Parameter(dt.random.random(k_shape) * 2 - 1)
    
    convolved = func(x, k, stride, padding)
    convolved.backward()
    x_grad, k_grad = x.grad, k.grad
    
    x_grad_approx = approx_grad(x.data.flatten(), lambda a: func(a.reshape(X_shape), k.data, stride, padding).data.sum())
    k_grad_approx = approx_grad(k.data.flatten(), lambda a: func(x.data, a.reshape(k_shape), stride, padding).data.sum()) 

    assert dt.allclose(k_grad.flatten(), k_grad_approx, atol=1e-6), f'Filter grad wrong'
    assert dt.allclose(x_grad.flatten(), x_grad_approx, atol=1e-6), f'Input grad wrong'
    print("OK\n")


X_shapes = [(1, 1, 7, 7), (2, 5, 3, 3), (3, 15, 1, 1)]
k_shapes = [(1, 13, 4, 4), (5, 8, 1, 1), (15, 20, 5, 5)]
strides  = [1, 2, 3]
paddings = [0, 1, 2]

for i, args in enumerate(zip(X_shapes, k_shapes, strides, paddings)):
    
    if i < 2: continue
    
    print(*args)
    check(*args)