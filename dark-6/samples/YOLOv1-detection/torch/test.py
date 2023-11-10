from config import *
from model import *
from utils import *
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 
import cv2

def predict(model, im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    im = A.Resize(IM_SIZE, IM_SIZE)(image=im)["image"]
    im = A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image=im)["image"]
    im = ToTensorV2()(image=im)["image"]
    im = im.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(im)

    pred = pred.to("cpu")
    bboxes = boxes_from_prediction(pred)
    return bboxes
    
def main():
    im = cv2.imread("db/images/000009.jpg")
    model = torch.load("model.pt", map_location=device)
    
    model.eval()
    bboxes = predict(model, im)
    plot_boxes(im, bboxes)

    cv2.imshow("Image", im)
    cv2.waitKey()


if __name__ == "__main__":
    main()