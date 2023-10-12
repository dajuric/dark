import numpy as np
from ttictoc import tic,toc
import dark

im = np.random.random((64, 3, 256, 256))
k  = np.random.random((1, 3, 5, 5))

dark.max_pool2d(im, 2)

tic()
res = dark.max_pool2d(im, 2)
print(toc())