import numpy as np
import random

random.seed(0)
np.random.seed(0)

im = np.random.randint(-16, +16, size=(5, 3, 2, 2)).astype(np.float32)
 
#---------------------------
import torch as tt
import torch.nn as nn

st = tt.tensor(im, requires_grad=True)

tRes = nn.functional.max_pool2d(st, kernel_size=2)
tRes = tt.sum(tRes)

tRes.backward()
#print(kt.grad)


#---------------------------
import dark
sd = dark.Parameter(im)
dRes = dark.max_pool2d(sd, kernel_size=2)
dRes = dark.sum(dRes)

dRes.backward()
#print(kd.grad)


print("Result equal? " + str(np.allclose(dRes.data.get(), tRes.detach().numpy(), rtol=0, atol=1e-3)))
print("Grad S equal? " + str(np.allclose(sd.grad.get(),   st.grad,               rtol=0, atol=1e-3)))
print("Done")