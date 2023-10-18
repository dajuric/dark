import numpy as np
import random
import time

random.seed(10)
np.random.seed(10)

im = np.random.randint(0, 255, size=(256, 3, 32, 32)).astype(np.float32)

#---------------------------
import dark

tic = time.time()

sd = dark.Parameter(im)
dRes = dark.max_pool2d(sd, kernel_size = 2)
dRes = dark.sum(dark.sum(dark.sum(dark.sum(dRes, dim = 0), dim = 1), dim = 2), dim = 3)
dRes.backward()

toc = time.time()
print(f"Elapsed: {int((toc - tic) * 1000)}ms")