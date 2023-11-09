from config import *
from model import YOLONano
from utils import *
import torch
import cv2
from glob import glob
import random

sAnchors = get_scaled_anchors()

def predict(model, im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    im = test_transforms(image=im, bboxes=[])["image"]
    im = im.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(im)
        pred = [x.cpu() for x in pred]

    bboxes = [
                *cell_to_boxes(pred[0], S[0], sAnchors=sAnchors[0]),
                *cell_to_boxes(pred[1], S[1], sAnchors=sAnchors[1]),
                *cell_to_boxes(pred[2], S[2], sAnchors=sAnchors[2])
             ]

    return bboxes
    
def main():
    im_files = sorted(glob(f"{DB_PATH}/val/**/*.jpg", recursive=True))
    im_file = random.choice(im_files)
    
    im = cv2.imread(im_file)  
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()

    bboxes = predict(model, im)
    plot_boxes(im, bboxes)

    cv2.imshow("Image", im)
    cv2.waitKey()


if __name__ == "__main__":
    main()