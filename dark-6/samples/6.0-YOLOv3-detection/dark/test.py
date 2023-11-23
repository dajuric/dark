import dark
from config import *
from model import BlazeFace as YoloNet
from utils import *
import cv2
from glob import glob
import random
import pickle

sAnchors = get_scaled_anchors()

def predict(model, im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    im = test_transforms(image=im, bboxes=[])["image"]
    im = np.expand_dims(im, 0)
    pred = model(im)
    
    bboxes = [
                *cell_to_boxes(pred[0].data, S[0], sAnchors=sAnchors[0]),
                *cell_to_boxes(pred[1].data, S[1], sAnchors=sAnchors[1])
             ]

    return bboxes
    
def main():
    im_files = [ "/hdd1/djuric/Desktop/db/celebA/img_celeba/000551.jpg" ] #sorted(glob(f"{DB_PATH}/train/**/*.jpg", recursive=True))
    im_file = random.choice(im_files)
    
    im = cv2.imread(im_file); im = cv2.resize(im, None, fx=0.5, fy=0.5)  
    model = pickle.load(open(MODEL_PATH, "rb"))
    model.eval()

    bboxes = predict(model, im)
    plot_boxes(im, bboxes)

    cv2.imshow("Image", im)
    cv2.waitKey()


if __name__ == "__main__":
    main()