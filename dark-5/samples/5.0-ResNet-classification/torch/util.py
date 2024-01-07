import numpy as np
import cv2
import torch
from config import *

def _draw_label(im, correct, label):
    color = (0, 255, 0) if correct == 1 else (0, 0, 255)
    cv2.putText(im, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def save_samples(dataset, model, str_labels, filename, grid = (3, 5)):
    h, w = grid
    i = 0
    
    image_rows = []
    for r in range(h):
        
        image_row = []
        for c in range(w):
            im, target = dataset[i]

            im_t = torch.unsqueeze(im.to(device), 0)
            prediction = model(im_t).detach().cpu().numpy()
           
            im = (im.numpy() * 127 + 127).astype(np.uint8)
            im = cv2.cvtColor(np.moveaxis(im, 0, 2), cv2.COLOR_RGB2BGR)
            im = np.ascontiguousarray(im)
            im = cv2.resize(im, (128, 128))

            is_correct = np.argmax(prediction) == np.argmax(target)
            _draw_label(im, is_correct, str_labels[np.argmax(prediction)])

            image_row.append(im)
            i += 1
            
        image_rows.append(np.concatenate(image_row, axis=1))
    
    table = np.concatenate(image_rows, axis=0)
    cv2.imwrite(filename, table)