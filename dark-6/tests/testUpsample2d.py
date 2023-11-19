import numpy as np
import random
import os

random.seed(0)
np.random.seed(0)

im = np.random.randint(-16, +16, size=(1, 3, 2, 2)).astype(np.float64)
 
#---------------------------
import torch as tt
import torch.nn as nn

st = tt.tensor(im, requires_grad=True)

tRes = nn.functional.interpolate(st, scale_factor=3, mode='nearest')
tRes = tt.sum(tRes)

tRes.backward()
#print(kt.grad)


#---------------------------
os.environ["USE_CPU"] = "True"

import dark
import dark.tensor as dt

sd = dark.Parameter(im)
dRes = dark.upsample2d(sd, factor=3)
dRes = dark.sum(dRes)

dRes.backward()
#print(kd.grad)


print("Result equal? " + str(np.allclose(dt.numpy(dRes.data), tRes.detach().numpy(), rtol=0, atol=1e-3)))
print("Grad X equal? " + str(np.allclose(dt.numpy(sd.grad),   st.grad,               rtol=0, atol=1e-3)))
print("Done")