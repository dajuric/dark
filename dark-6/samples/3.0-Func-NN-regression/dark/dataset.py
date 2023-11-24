import random
import numpy as np

def get_data():
    def f(x, y):
        return np.sin(np.sqrt(x ** 2 + y ** 2))

    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    X = X.reshape(-1, 1); Y = Y.reshape(-1, 1)
    data = np.concatenate((X, Y), axis=1)
    values = Z.reshape(-1, 1)
    return data, values

def normalize_data(x):
    min_val = x.min(axis = 0)
    max_val = x.max(axis = 0)
    
    x_norm = (x - min_val) / (max_val - min_val)
    return x_norm