import torch
import torch.nn as nn
import cv2

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.zeros_(m.bias.data)

    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.zeros_(m.bias.data)

def save_samples(batch, filename, grid = (3, 5)):
    h, w = grid
    i = 0
    
    image_rows = []
    for r in range(h):
        
        image_row = []
        for c in range(w):
            
            im = batch[i]
            im = torch.sigmoid(im)
            
            im = (im.squeeze() * 255).type(torch.uint8)
            im = torch.moveaxis(im, 0, 2)
            
            image_row.append(im)
            i += 1
            
        image_rows.append(torch.concatenate(image_row, dim=1))
    
    table = torch.concatenate(image_rows, dim=0)
    cv2.imwrite(filename, table.cpu().numpy())