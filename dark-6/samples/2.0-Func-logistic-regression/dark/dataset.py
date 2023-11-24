import random
import numpy as np

def get_data():
    X = []; Y = []

    for i in range(100):
        x_1 = random.random()
        x_2 = random.random()
        
        y = 1 if x_1 < 0.5 else 0 #split

        X.append((x_1, x_2))
        Y.append([y])

    X = np.array(X); Y = np.array(Y)
    return X, Y