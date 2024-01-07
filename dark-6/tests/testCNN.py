import numpy as np
import random

random.seed(1)
np.random.seed(1)

X = np.random.randint(-16, +16, size=(1, 3, 9, 9)).astype(np.float32)
# -----------------------------------------------------------------------

#import os
#os.environ["USE_CPU"] = "True"

import dark
import dark.nn as nn
import dark.optim as optim
import dark.tensor as dt

random.seed(2)
np.random.seed(2)
dt.random.seed(2)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(3, 1, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
                            )

    def forward(self, x):
        x = self.block(x)
        return x

def apply(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        m.weights.data = dt.ones_like(m.weights.data) * 0.001
        m.bias.data = dt.ones_like(m.bias.data) * 0.001

    if isinstance(m, nn.Linear):
        m.weights.data = dt.ones_like(m.weights.data) * 0.001
        m.bias.data = dt.ones_like(m.bias.data) * 0.001


model = CNN()
model.train()
model.apply(apply)   
sgd = optim.SGD(model.parameters(), lr=0.001)

for _ in range(1):
    sgd.zero_grad()
    result = model(X)
    result = dark.sum(result)
    result.backward()
    sgd.step()
    #print(result)

dResult = result
for p in model.parameters():
    print(p.grad)
    print()
    
print("----" * 25)
#---------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim

random.seed(2)
np.random.seed(2)
dt.random.seed(dt.uint64(2))

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(3, 1, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
                            )

    def forward(self, x):
        x = self.block(x)
        return x

def apply(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        m.weight = nn.Parameter(torch.ones_like(m.weight) * 0.001)
        m.bias = nn.Parameter(torch.ones_like(m.bias) * 0.001)

    if isinstance(m, nn.Linear):
        m.weight = nn.Parameter(torch.ones_like(m.weight) * 0.001)
        m.bias = nn.Parameter(torch.ones_like(m.bias) * 0.001)

model = CNN()
model.train()
model.apply(apply)
sgd = optim.SGD(model.parameters(), lr=0.001)

for _ in range(1):
    sgd.zero_grad()
    result = model(torch.tensor(X))
    result = torch.sum(result)
    result.backward()
    sgd.step()
    #print(result)

tResult = result
for p in model.parameters():
    print(p.grad)
    print()

print("Result equal? " + str(np.allclose(dResult.data, tResult.cpu().detach().numpy(), rtol=0, atol=1e-3)))
print()