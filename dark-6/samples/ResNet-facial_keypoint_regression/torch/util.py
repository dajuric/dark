import torch
import cv2
import numpy as np

def _draw_keypoints(im, keypoints, color):
    imH, imW, _ = im.shape

    for x, y in keypoints.reshape(-1, 2):
        x = int(x * imW)
        y = int(y * imH)

        cv2.circle(im, (x, y), 3, color, -1)

def save_samples(dataset, model, filename, grid = (3, 5), device = "cpu"):
    h, w = grid
    i = 0
    
    image_rows = []
    for r in range(h):
        
        image_row = []
        for c in range(w):
            im, target = dataset[i]
            prediction = model(im.unsqueeze(0).to(device)).cpu().numpy()

            im = (im * 127 + 127).type(torch.uint8).numpy()
            im = np.ascontiguousarray(np.moveaxis(im, 0, 2))

            _draw_keypoints(im, target, (255, 255, 0))
            _draw_keypoints(im, prediction, (0, 255, 0))

            image_row.append(im)
            i += 1
            
        image_rows.append(np.concatenate(image_row, axis=1))
    
    table = np.concatenate(image_rows, axis=0)
    cv2.imwrite(filename, table)