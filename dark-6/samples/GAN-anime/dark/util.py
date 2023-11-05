import dark.tensor as dt
import cv2

def save_samples(batch, filename, grid = (3, 5)):
    h, w = grid
    i = 0
    
    image_rows = []
    for r in range(h):
        
        image_row = []
        for c in range(w):
            
            im = batch[i]
            den = dt.add(1, dt.exp(dt.negative(im)))
            im = dt.divide(1, den)
            
            im = (im.squeeze() * 255).astype(dt.uint8)
            im = dt.rollaxis(im, 0, 3)
            
            image_row.append(im)
            i += 1
            
        image_rows.append(dt.concatenate(image_row, axis=1))
    
    table = dt.concatenate(image_rows, axis=0)
    cv2.imwrite(filename, table.get())