import numpy as np
import cv2
import dark.tensor as dt

def _draw_label(im, correct):
    color = (0, 255, 0) if correct == 1 else (0, 0, 255)
    cv2.circle(im, (25, 25), 5, color, -1)

def save_samples(dataset, model, filename, grid = (3, 5)):
    h, w = grid
    i = 0
    
    image_rows = []
    for r in range(h):
        
        image_row = []
        for c in range(w):
            im, target = dataset[i]
            prediction = model(np.expand_dims(im, 0)).data

            im = (im * 127 + 127).astype(np.uint8)
            im = cv2.cvtColor(np.moveaxis(im, 0, 2), cv2.COLOR_GRAY2BGR)
            im = np.ascontiguousarray(im)
            im = cv2.resize(im, (64, 64))

            is_correct = np.argmax(prediction) == np.argmax(target)
            _draw_label(im, is_correct)

            image_row.append(im)
            i += 1
            
        image_rows.append(np.concatenate(image_row, axis=1))
    
    table = np.concatenate(image_rows, axis=0)
    cv2.imwrite(filename, table)