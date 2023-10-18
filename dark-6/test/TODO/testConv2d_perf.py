import numpy as np
import random
import time

random.seed(0)
np.random.seed(0)

im = np.random.randint(-16, +16, size=(128, 3, 32, 32)).astype(np.float32)
k  = np.random.randint(-16, +16, size=(32,  3,  5,  5)).astype(np.float32)
 
#---------------------------
import dark

sd = dark.Parameter(im)
kd = dark.Parameter(k)
dRes = dark.conv2d(sd, kd, padding=2)
dRes = dark.sum(dark.sum(dark.sum(dark.sum(dRes, dim = 0), dim = 1), dim = 2), dim = 3)
dRes.backward()

print("------------------------------------------------------")
tic = time.time()

sd = dark.Parameter(im)
kd = dark.Parameter(k)
dRes = dark.conv2d(sd, kd, padding=2)
dRes = dark.sum(dark.sum(dark.sum(dark.sum(dRes, dim = 0), dim = 1), dim = 2), dim = 3)
dRes.backward()

toc = time.time()
print(f"Elapsed: {int((toc - tic) * 1000)}ms")


