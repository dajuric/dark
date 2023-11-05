import torch
import cv2

def save_samples(batch, filename, grid = (3, 5)):
    h, w = grid
    i = 0
    
    image_rows = []
    for r in range(h):
        
        image_row = []
        for c in range(w):
            
            im = batch[i]
            den = torch.add(1, torch.exp(torch.negative(im)))
            im = torch.divide(1, den)
            
            im = (im.squeeze() * 255).type(torch.uint8)
            im = torch.moveaxis(im, 0, 2)
            
            image_row.append(im)
            i += 1
            
        image_rows.append(torch.concatenate(image_row, dim=1))
    
    table = torch.concatenate(image_rows, dim=0)
    cv2.imwrite(filename, table.cpu().numpy())