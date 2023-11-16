from dark.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from config import *
from utils import *
import glob

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
            sTargets = np.zeros((NUM_ANCHORS, S[sIdx], S[sIdx], 4+1+1))
            
            for box in boxes:
                self._assign_cell(box, sTargets, sAnchors, S[sIdx])

            targets.append(sTargets)

        return im, *targets


    def _read_boxes(self, lblFile):
        boxes = np.loadtxt(lblFile, delimiter=" ", ndmin=2, dtype=np.float64)
        boxes = np.roll(boxes, -1)
        boxes = boxes.tolist()

        return boxes

    def _assign_cell(self, box, targets, anchors, s):
        x, y, w, h, lbl = box
        iou_anchors = self._iou_wh(box[2:4], anchors)
        aIndices = iou_anchors.argsort(axis=0)[::-1]
        scale_has_anchor = False

        for aIdx in aIndices:
            r, c = int(s * y), int(s * x)

            if scale_has_anchor and iou_anchors[aIdx] > 0.5:
                targets[aIdx, r, c, 0] = -1
                continue

            anchor_taken = targets[aIdx, r, c, 0]
            if scale_has_anchor or anchor_taken: 
                continue
            
            x_cell, y_cell = s * x - c, s * y - r
            w_cell, h_cell = s * w,     s * h
            box_coords = [x_cell, y_cell, w_cell, h_cell]

            targets[aIdx, r, c, 0] = 1
            targets[aIdx, r, c, 1:5] = np.array(box_coords)
            targets[aIdx, r, c, 5] = int(lbl)
            
            scale_has_anchor = True
            
    def _iou_wh(self, box, anchors):
        ious = []
        for a in anchors:
            intersect = min(box[0], a[0]) *  min(box[1], a[1])
            union = box[0] * box[1] + a[0] * a[1] - intersect
            iou = intersect / union
            ious.append(iou)

        return np.array(ious)
    
def get_dataloaders():
    dbTrain = YoloDataset(f"{DB_PATH}/", train_transforms)
    dbTest  = YoloDataset(f"{DB_PATH}/", test_transforms) 

    trainLoader = DataLoader(dbTrain, BATCH_SIZE, shuffle=True)
    testLoader  = DataLoader(dbTest,  BATCH_SIZE, shuffle=False)

    print(f"Db samples: train {len(dbTrain)}, test {len(dbTest)}")
    return trainLoader, testLoader


if __name__ == "__main__":
    dataset = YoloDataset(f"{DB_PATH}/", test_transforms)

    im, target = dataset[382]
    boxes = []

    for sIdx in range(len(target)):
        s_boxes = cell_to_boxes(target[sIdx].unsqueeze(0), S[sIdx])
        boxes = [*boxes, *s_boxes]

    im = im.permute(1, 2, 0).numpy()
    plot_boxes(im, boxes)
    cv2.imshow("Test", im)
    cv2.waitKey()