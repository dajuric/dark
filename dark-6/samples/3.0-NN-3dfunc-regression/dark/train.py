import os
os.environ["USE_CPU"] = "True"

import dark
import dark.tensor as dt
import dark.nn as nn
import dark.optim as optim

import matplotlib.pyplot as plt
from model import get_net
from util import *
from dataset import *


X, y = get_data()
X = normalize_data(X); y = normalize_data(y)

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