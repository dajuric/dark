import os
os.environ["USE_CPU"] = "True"

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

def check(X_shape, k_shape, stride, padding):
    x = dark.Parameter(dt.random.random(X_shape) * 2 - 1)
    k = dark.Parameter(dt.random.random(k_shape) * 2 - 1)
    
    convolved = dark.conv2d(x, k, stride, padding)
    convolved.backward()
    x_grad, k_grad = x.grad, k.grad
    
    x_grad_approx = approx_grad(x.data.flatten(), lambda a: dark.conv2d(a.reshape(X_shape), k.data, stride, padding).data.sum())
    k_grad_approx = approx_grad(k.data.flatten(), lambda a: dark.conv2d(x.data, a.reshape(k_shape), stride, padding).data.sum()) 

    assert dt.allclose(k_grad.flatten(), k_grad_approx, atol=1e-6), f'Filter grad wrong'
    assert dt.allclose(x_grad.flatten(), x_grad_approx, atol=1e-6), f'Input grad wrong'
    print("OK\n")


X_shapes = [(1, 1, 28, 28), (2, 5, 27, 27), (3, 15, 31, 31)]
k_shapes = [(32, 1, 4, 4), (11, 5, 19, 19), (32, 15, 3, 3)]
strides  = [1, 2, 3]
paddings = [0, 1, 2]

for args in zip(X_shapes, k_shapes, strides, paddings):
    print(*args)
    check(*args)