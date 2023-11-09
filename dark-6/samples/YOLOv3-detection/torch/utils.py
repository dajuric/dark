import torch
import cv2
from config import *

def iou(boxesA, boxesB):
    boxA_x1 = boxesA[..., 0:1] - boxesA[..., 2:3] / 2
    boxA_y1 = boxesA[..., 1:2] - boxesA[..., 3:4] / 2
    boxA_x2 = boxesA[..., 0:1] + boxesA[..., 2:3] / 2
    boxA_y2 = boxesA[..., 1:2] + boxesA[..., 3:4] / 2

    boxB_x1 = boxesB[..., 0:1] - boxesB[..., 2:3] / 2
    boxB_y1 = boxesB[..., 1:2] - boxesB[..., 3:4] / 2
    boxB_x2 = boxesB[..., 0:1] + boxesB[..., 2:3] / 2
    boxB_y2 = boxesB[..., 1:2] + boxesB[..., 3:4] / 2

    x1 = torch.max(boxA_x1, boxB_x1)
    y1 = torch.max(boxA_y1, boxB_y1)
    x2 = torch.min(boxA_x2, boxB_x2)
    y2 = torch.min(boxA_y2, boxB_y2)
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    boxA_area = abs((boxA_x2 - boxA_x1) * (boxA_y2 - boxA_y1))
    boxB_area = abs((boxB_x2 - boxB_x1) * (boxB_y2 - boxB_y1))

    ious = intersection / (boxA_area + boxB_area - intersection + 1e-6)
    return ious


def get_scaled_anchors():
    anchors = []

    for sIdx, s in enumerate(S):
        sAnchors = []
        anchors.append(sAnchors)

        for aW, aH in ANCHORS[sIdx]:
            sAnchors.append((aW * s, aH * s))

    return torch.tensor(anchors)

def cell_to_boxes(sPreds, s, threshold = 0.5, sAnchors = None):
    scores = sPreds[..., 0:1]
    boxes  = sPreds[..., 1:5]

    if sAnchors is None:
        labels = sPreds[..., 5:6]
    else:  
        scores = torch.sigmoid(scores)
        labels = torch.argmax(sPreds[..., 5:], dim=-1).unsqueeze(-1)

        sAnchors = sAnchors.reshape(1, NUM_ANCHORS, 1, 1, 2)
        boxes[..., 0:2] = torch.sigmoid(boxes[..., 0:2])
        boxes[..., 2:4] = torch.exp(boxes[..., 2:4]) * sAnchors


    batch_size = sPreds.shape[0]
    bboxes = []

    for batchIdx in range(batch_size):
        for aIdx in range(NUM_ANCHORS):
            for cellRow in range(s):
                for cellCol in range(s):
                    p = scores[batchIdx, aIdx, cellRow, cellCol].item()
                    if p < threshold:
                        continue

                    cellX, cellY, cellW, cellH = boxes[batchIdx, aIdx, cellRow, cellCol, ...].numpy()
                    x = 1 / s * (cellX + cellCol)
                    y = 1 / s * (cellY + cellRow)
                    w = 1 / s * (cellW)
                    h = 1 / s * (cellH)

                    lbl = labels[batchIdx, aIdx, cellRow, cellCol].item()
                    bb = [x, y, w, h, p, lbl]
                    bboxes.append(bb)

    return bboxes

def plot_boxes(im, boxes):
    colors = (torch.rand((C, 3)) * 255).int()
    imH, imW, _ = im.shape

    for bb in boxes:
        bxC, byC, bw, bh, p, lbl = bb
        x = int((bxC - bw / 2) * imW)
        y = int((byC - bh / 2) * imH)
        w = int(bw * imW)
        h = int(bh * imH)

        color = colors[int(lbl)].numpy().tolist()
        cv2.rectangle(im, (x, y), (x + w, y + h), color, 3)

if __name__ == "__main__":
    sa = get_scaled_anchors()
    print(sa)