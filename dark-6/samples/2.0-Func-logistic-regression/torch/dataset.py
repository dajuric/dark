import random
import torch

def get_data():
    X = []; Y = []

    for i in range(100):
        x_1 = random.random()
        x_2 = random.random()
        
        y = 1 if x_1 < 0.5 else 0 #split

        X.append((x_1, x_2))
        Y.append([y])

    X = torch.tensor(X, dtype=torch.float64)
    Y = torch.tensor(Y, dtype=torch.float64)
    return X, Y