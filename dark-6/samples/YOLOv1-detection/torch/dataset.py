import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 
from config import *

class VOCDataset(Dataset):
    def __init__(self, 
                csv_file, img_dir="db/images/", label_dir="db/labels/",
                transform=None):

        self.img_dir = img_dir
        self.label_dir = label_dir

        self.transform = transform
        self.annotations = np.genfromtxt(csv_file, skip_header=True, delimiter=',', dtype=str)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self,  index):
        label_path = os.path.join(self.label_dir, self.annotations[index][1])
        boxes = self._read_boxes(label_path)
        #boxes = self._boxes_to_squares(boxes) #as B=1, we have to have all the boxes of the same size

        img_path = os.path.join(self.img_dir, self.annotations[index][0])
        image = np.asanyarray(Image.open(img_path))

        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes)
            image = transformed["image"]
            boxes = transformed["bboxes"]

        label_matrix = self._convert_to_cells(boxes)
        return image, label_matrix

    def _read_boxes(self, label_path):
        boxes = []

        with open(label_path) as f:
            for row in f.readlines():
                lbl, x, y, w, h = map(float, row.split())
                boxes.append([x, y, w, h, int(lbl)])

        return boxes

    def _boxes_to_squares(self, boxes):
        squares = []

        for i in range(len(boxes)):
            x, y, w, h, lbl = boxes[i]
            maxDim = max(w, h)

            maxDim = min(maxDim, (1 - x) * 2, (x - 0) * 2)
            maxDim = min(maxDim, (1 - y) * 2, (y - 0) * 2)

            squares.append([x, y, maxDim, maxDim, lbl])

        return squares

    def _convert_to_cells(self, boxes):
        label_matrix = torch.zeros((S, S, C + (1+4) * 1))

        for box in boxes:
            x, y, w, h, lbl = box

            cell_row, cell_col = int(S * y), int(S * x)
            if label_matrix[cell_row, cell_col, C] == 1:
                continue

            x_cell = S * x - int(S * x)
            y_cell = S * y - int(S * y)
            w_cell = w * S
            h_cell = h * S
            cell_box = [x_cell, y_cell, w_cell, h_cell]

            label_matrix[cell_row, cell_col, C] = 1
            label_matrix[cell_row, cell_col, (C+1):(C+1+4)] = torch.tensor(cell_box)
            label_matrix[cell_row, cell_col, lbl] = 1

        return label_matrix

    
def get_dataloaders():
    tr_transforms = A.Compose([
        A.Resize(IM_SIZE, IM_SIZE),
        A.HorizontalFlip(),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ], 
    bbox_params=A.BboxParams(format="yolo"))

    te_transforms = A.Compose([
        A.Resize(IM_SIZE, IM_SIZE),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ], 
    bbox_params=A.BboxParams(format="yolo"))

    tr_dataset = VOCDataset("db/8examples.csv", transform=tr_transforms)
    te_dataset = VOCDataset("db/8examples.csv",   transform=te_transforms)

    tr_loader = DataLoader(tr_dataset, BATCH_SIZE, shuffle=True)
    te_loader = DataLoader(te_dataset, BATCH_SIZE, shuffle=True)

    return tr_loader, te_loader


if __name__ == "__main__":
    from utils import *
    SAMPLE_IDX = 1

    tr_loader, _ = get_dataloaders()
    tr_batch = next(iter(tr_loader)) 
    imBatch, bbMatBatch = tr_batch

    im = imBatch[SAMPLE_IDX, ...]; bbMat = bbMatBatch[SAMPLE_IDX, ...].unsqueeze(0)
    im = im.permute(1, 2, 0).numpy()
    im = np.ascontiguousarray(im)

    boxes = boxes_from_prediction(bbMat)
    plot_boxes(im, boxes)

    cv2.imshow("Image", im)
    cv2.waitKey()
