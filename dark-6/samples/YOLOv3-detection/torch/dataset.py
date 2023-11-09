import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from config import *
from utils import *
import glob
import random

class YoloDataset(Dataset):
    def __init__(self, im_folder, imT=None):
        super().__init__()
        self.lblFiles = sorted(glob.glob(f"{im_folder}/**/*.txt", recursive=True))
        self.imT = imT

    def __len__(self):
        return len(self.lblFiles)

    def __getitem__(self, index):
        lblFile = self.lblFiles[index]
        imFile  = lblFile.replace(".txt", ".jpg")

        im = np.array(Image.open(imFile).convert("RGB"))
        boxes = self._read_boxes(lblFile)

        if self.imT:
            augments = self.imT(image=im, bboxes=boxes)
            im = augments["image"]
            boxes = augments["bboxes"]

        targets = []
        for sIdx in range(len(S)):
            sAnchors = ANCHORS[sIdx]
            sTargets = torch.zeros(NUM_ANCHORS, S[sIdx], S[sIdx], 4+1+1)
            
            for box in boxes:
                self._assign_cell(box, sTargets, sAnchors, S[sIdx])

            targets.append(sTargets)

        return im, tuple(targets)


    def _read_boxes(self, lblFile):
        boxes = np.loadtxt(lblFile, delimiter=" ", ndmin=2, dtype=np.float64)
        boxes = np.roll(boxes, -1)
        boxes = boxes.tolist()

        return boxes

    def _assign_cell(self, box, targets, anchors, s):
        x, y, w, h, lbl = box
        iou_anchors = self._iou_wh(box[2:4], anchors)
        aIdx = np.argmax(iou_anchors)

        r, c = int(s * y), int(s * x)
        anchor_taken = targets[aIdx, r, c, 0]
        if anchor_taken: 
            return

        x_cell, y_cell = s * x - c, s * y - r
        w_cell, h_cell = s * w,     s * h
        box_coords = [x_cell, y_cell, w_cell, h_cell]

        targets[aIdx, r, c, 0] = 1
        targets[aIdx, r, c, 1:5] = torch.tensor(box_coords)
        targets[aIdx, r, c, 5] = int(lbl)
            
    def _iou_wh(self, box, anchors):
        ious = []
        for a in anchors:
            intersect = min(box[0], a[0]) *  min(box[1], a[1])
            union = box[0] * box[1] + a[0] * a[1] - intersect
            iou = intersect / union
            ious.append(iou)

        return np.array(ious)
    
def get_dataloaders():
    dbTrain = YoloDataset(f"{DB_PATH}/train/", train_transforms)
    dbTest  = YoloDataset(f"{DB_PATH}/val/",   test_transforms) 

    workers = 0
    trainLoader = DataLoader(dbTrain, BATCH_SIZE, shuffle=True,  num_workers=workers)
    testLoader  = DataLoader(dbTest,  BATCH_SIZE, shuffle=False, num_workers=workers)

    print(f"Db samples: train {len(dbTrain)}, test {len(dbTest)}")
    return trainLoader, testLoader


if __name__ == "__main__":
    dataset = YoloDataset(f"{DB_PATH}/val/", test_transforms)

    im, target = dataset[10]
    boxes = []

    for sIdx in range(len(target)):
        s_boxes = cell_to_boxes(target[sIdx].unsqueeze(0), S[sIdx])
        boxes = [*boxes, *s_boxes]

    im = im.permute(1, 2, 0).numpy()
    plot_boxes(im, boxes)
    cv2.imshow("Test", im)
    cv2.waitKey()