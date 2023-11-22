import numpy as np
import cv2

def _draw_label(im, prediction):
    txt = str(prediction)
    cv2.putText(im, txt, (5, 5), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2)

def save_samples(dataset, model, filename, grid = (3, 5)):
    h, w = grid
    i = 0
    
    image_rows = []
    for r in range(h):
        
        image_row = []
        for c in range(w):
            im, target = dataset[i]
            prediction = model(np.expand_dims(im, 0)).detach().cpu().numpy()

            im = (im * 127 + 127).astype(np.uint8)
            im = cv2.cvtColor(np.moveaxis(im, 0, 2), cv2.COLOR_RGB2BGR)
            im = np.ascontiguousarray(im)

            _draw_label(im, np.argmax(prediction))

            image_row.append(im)
            i += 1
            
        image_rows.append(np.concatenate(image_row, axis=1))
    
    table = np.concatenate(image_rows, axis=0)
    cv2.imwrite(filename, table)