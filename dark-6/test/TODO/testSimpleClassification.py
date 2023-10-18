import dark
import dark.nn as nn
from dark.optim import SGD

# import torch
# import torch.nn as nn
# from torch.optim import *

import numpy as np
import random

CLASS_COUNT = 2
EPOCHS = 1000
random.seed(0)

def get_data():
    X = []; Y = []

    for i in range(50):
        x_1 = random.random()
        x_2 = random.random()
        #y = 0 if x_1 < 0.5 else 1 #split
        y = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0 #xor

        X.append((x_1, x_2))
        Y.append([0, 0]); Y[-1][y] = 1

    X = np.array(X, dtype=np.float32); Y = np.array(Y, dtype=np.float32).reshape(-1, CLASS_COUNT)
    X = (X - X.mean(axis=0)) / X.std()
    return X, Y
       
class PointNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),

            nn.Linear(5, CLASS_COUNT)        
        )

    def forward(self, x):
        logits = self.network(x)
        return logits

def train_loop(data, model: PointNetwork, loss_fn: nn.CrossEntropyLoss, optimizer: SGD):
    model.train()

    X, y = data; #X = torch.tensor(X); y = torch.tensor(y)
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss = loss.value.item()
    print(f"loss: {loss:>7f}")
 
def main():
    data = get_data()
    model = PointNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.3)

    for e in range(EPOCHS):
        print(f"Epoch {e+1}\n-------------------------------")
        train_loop(data, model, loss_fn, optimizer)

    print("Done!")


if __name__ == "__main__":
    main()

