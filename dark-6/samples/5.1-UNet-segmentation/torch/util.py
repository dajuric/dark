import cv2
import torch
from config import *

def save_samples(dataset, model, filename, count = 5):
    image_rows = []

    for i in range(count):
        im, true_mask = dataset[i]
        im = torch.unsqueeze(im, 0).type(torch.float32)

        pred_mask = model(im.to(device)).detach().cpu()
        pred_mask = torch.sigmoid(pred_mask)

        im        = (im.squeeze()        * 255).type(torch.uint8)
        pred_mask = (pred_mask.squeeze() * 255).type(torch.uint8)
        true_mask = (true_mask.squeeze() * 255).type(torch.uint8)

        row = torch.concatenate([im, torch.stack([true_mask] * 3, axis=0), torch.stack([pred_mask] * 3, axis=0)], axis=1)
        row = torch.moveaxis(row, 0, 2)
        image_rows.append(row)

    image_table = torch.concatenate(image_rows, axis=1)
    cv2.imwrite(filename, image_table.detach().numpy())