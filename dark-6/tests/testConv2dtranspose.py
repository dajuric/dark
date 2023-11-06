import numpy as np
import random

random.seed(0)
np.random.seed(0)

im = np.random.randint(-16, +16, size=(4,  1, 32, 32)).astype(np.float32)
k  = np.random.randint(-16, +16, size=(2,  1,  3,  3)).astype(np.float32); k = k.transpose(1, 0, 2, 3)
 
#---------------------------
import torch as tt
import torch.nn as nn

st = tt.tensor(im, requires_grad=True)
kt = tt.tensor(k, requires_grad=True)

tRes = nn.functional.conv_transpose2d(st, kt, padding=(0, 0), stride=(2, 2))
tRes = tt.sum(tRes)

tRes.backward()
#print(kt.grad)


#---------------------------
import dark
sd = dark.Parameter(im)
kd = dark.Parameter(k)
dRes = dark.conv_transpose2d(sd, kd, padding=0, stride=2)
dRes = dark.sum(dRes)

dRes.backward()
#print(kd.grad)


print("Result equal? " + str(np.allclose(dRes.data.get(), tRes.detach().numpy(), rtol=0, atol=1e-3)))
print("Grad K equal? " + str(np.allclose(kd.grad.get(),   kt.grad,               rtol=0, atol=1e-3)))
print("Grad S equal? " + str(np.allclose(sd.grad.get(),   st.grad,               rtol=0, atol=1e-3)))
print("Done")