import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from model import get_net
from util import *
from dataset import *


X, y = get_data()
X = normalize_data(X);                      y = normalize_data(y)
X = torch.tensor(X, dtype=torch.float32);   y = torch.tensor(y, dtype=torch.float32)

model = get_net()
model.train()

loss_fn = nn.MSELoss()
opt  = optim.Adam(model.parameters(), lr=0.03)

for i in range(2000):
   estimates  = model(X)
   loss = loss_fn(estimates, y)

   opt.zero_grad()
   loss.backward()
   opt.step()
      
   if i % 50 == 0:   
      print(loss)
      plot(X, y, lambda x: model(x).data)

plt.show(block=True)