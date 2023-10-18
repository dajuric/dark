import numpy as np
import random

random.seed(0)
np.random.seed(0)

im = np.random.randint(-16, +16, size=(16, 3, 32, 32)).astype(np.float32)
k  = np.random.randint(-16, +16, size=(6,  3,  5,  5)).astype(np.float32)
 
#---------------------------
import dark
sd = dark.Parameter(im)
kd = dark.Parameter(k)
dRes = dark.conv2d(sd, kd, padding=2)
dRes = dark.sum(dark.sum(dark.sum(dark.sum(dRes, dim = 0), dim = 1), dim = 2), dim = 3)

dRes.backward()
#print(kd.grad)


#---------------------------
import torch
st = torch.tensor(im, requires_grad=True)
kt = torch.tensor(k, requires_grad=True)

tRes = torch.conv2d(st, kt, padding=2)
tRes = torch.sum(torch.sum(torch.sum(torch.sum(tRes, dim=0, keepdim=True), dim=1, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True)

tRes.backward()
#print(kt.grad)

print("Result equal? " + str(np.array_equal(dRes.value, tRes.detach().numpy())))
print("Grad K equal? " + str(np.array_equal(kd.grad,    kt.grad.numpy())))
print("Grad S equal? " + str(np.array_equal(sd.grad,    st.grad.numpy())))
print("Done")
