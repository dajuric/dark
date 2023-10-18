import numpy as np
import random

random.seed(10)
np.random.seed(10)

im = np.random.randint(0, 255, size=(32, 3, 32, 32)).astype(np.float32)

#---------------------------
import dark
sd = dark.Parameter(im)
dRes = dark.max_pool2d(sd, kernel_size = 2)
dRes = dark.sum(dark.sum(dark.sum(dark.sum(dRes, dim = 0), dim = 1), dim = 2), dim = 3)

dRes.backward()
#print(sd.grad)


#---------------------------
import torch
st = torch.tensor(im, requires_grad=True)
tRes = torch.max_pool2d(st, kernel_size=2)
tRes = torch.sum(torch.sum(torch.sum(torch.sum(tRes, dim=0, keepdim=True), dim=1, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True)

tRes.backward()
#print(st.grad)

print("Result equal? " + str(np.array_equal(dRes.value, tRes.detach().numpy())))
print("Grad S equal? " + str(np.array_equal(sd.grad,    st.grad.numpy())))
print("Done")
