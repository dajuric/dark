import dark.tensor as dt
import dark.nn as nn
from dark.nn.init import default_init_weights
import cv2

def init_weights(m):
    default_init_weights(m)

    if isinstance(m, nn.Conv2d):
        m.weights.data = dt.random.normal(0.0, 0.02, m.weights.data.shape)
        m.bias.data = dt.zeros_like(m.bias.data)

    if isinstance(m, nn.BatchNorm2d):
        m.gamma.data = dt.random.normal(1.0, 0.02, m.gamma.data.shape)
        m.beta.data = dt.zeros_like(m.beta.data)

def save_samples(batch, filename, grid = (3, 5)):
    h, w = grid
    i = 0
    
    image_rows = []
    for r in range(h):
        
        image_row = []
        for c in range(w):
            
            im = batch[i]
            im = dt.sigmoid(im)
            
            im = (im.squeeze() * 255).astype(dt.uint8)
            im = dt.rollaxis(im, 0, 3)
            
            image_row.append(im)
            i += 1
            
        image_rows.append(dt.concatenate(image_row, axis=1))
    
    table = dt.concatenate(image_rows, axis=0)
    cv2.imwrite(filename, dt.numpy(table))