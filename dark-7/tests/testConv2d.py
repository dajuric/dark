import numpy as np
import random
import os

random.seed(0)
np.random.seed(0)

im = np.random.randint(-16, +16, size=(2,  2, 9, 9)).astype(np.float64)
k  = np.random.randint(-16, +16, size=(3,  2,  3,  3)).astype(np.float64)
 
#---------------------------
import torch as tt
import torch.nn as nn

st = tt.tensor(im, requires_grad=True)
kt = tt.tensor(k, requires_grad=True)

tRes = nn.functional.conv2d(st, kt, padding=(0, 0), stride=(1, 1))
tRes = tt.sum(tRes)

tRes.backward()
#print(kt.grad)


#---------------------------
os.environ["USE_CPU"] = "True"

import dark
import dark.tensor as dt

sd = dark.Parameter(im)
kd = dark.Parameter(k)
dRes = dark.conv2d(sd, kd, padding=0, stride=1)
dRes = dark.sum(dRes)

dRes.backward()
#print(kd.grad)


print("Result equal? " + str(np.allclose(dt.numpy(dRes.data), tRes.detach().numpy(), rtol=0, atol=1e-3)))
print("Grad K equal? " + str(np.allclose(dt.numpy(kd.grad),   kt.grad,               rtol=0, atol=1e-3)))
print("Grad X equal? " + str(np.allclose(dt.numpy(sd.grad),   st.grad,               rtol=0, atol=1e-3)))
print("Done")