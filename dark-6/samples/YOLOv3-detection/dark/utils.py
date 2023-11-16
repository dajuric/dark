import dark.tensor as dt
import numpy as np
import cv2
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn, TaskProgressColumn
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

    x1 = dt.maximum(boxA_x1, boxB_x1)
    y1 = dt.maximum(boxA_y1, boxB_y1)
    x2 = dt.maximum(boxA_x2, boxB_x2)
    y2 = dt.maximum(boxA_y2, boxB_y2)

    intersection = dt.clip(x2 - x1, 0, None) * dt.clip(y2 - y1, 0, None)
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

    return dt.array(anchors)

def cell_to_boxes(sPreds, s, threshold = 0.5, sAnchors = None):
    scores = sPreds[..., 0:1]
    boxes  = sPreds[..., 1:5]

    if sAnchors is None:
        labels = sPreds[..., 5:6]
    else:  
        scores = dt.sigmoid(scores)
        labels = dt.expand_dims(dt.argmax(sPreds[..., 5:], axis=-1), -1)

        sAnchors = sAnchors.reshape(1, NUM_ANCHORS, 1, 1, 2)
        boxes[..., 0:2] = dt.sigmoid(boxes[..., 0:2])
        boxes[..., 2:4] = dt.exp(boxes[..., 2:4]) * sAnchors


    batch_size = sPreds.shape[0]
    bboxes = []

    for batchIdx in range(batch_size):
        for aIdx in range(NUM_ANCHORS):
            for cellRow in range(s):
                for cellCol in range(s):
                    p = scores[batchIdx, aIdx, cellRow, cellCol].item()
                    if p < threshold:
                        continue

                    cellX, cellY, cellW, cellH = boxes[batchIdx, aIdx, cellRow, cellCol, ...]
                    x = 1 / s * (cellX + cellCol)
                    y = 1 / s * (cellY + cellRow)
                    w = 1 / s * (cellW)
                    h = 1 / s * (cellH)

                    lbl = labels[batchIdx, aIdx, cellRow, cellCol].item()
                    bb = [x, y, w, h, p, lbl]
                    bboxes.append(bb)

    return bboxes

def plot_boxes(im, boxes):
    colors = (dt.random.rand(*(C, 3)) * 255).astype(dt.int32)
    imH, imW, _ = im.shape

    for bb in boxes:
        bxC, byC, bw, bh, p, lbl = bb
        x = int((bxC - bw / 2) * imW)
        y = int((byC - bh / 2) * imH)
        w = int(bw * imW)
        h = int(bh * imH)

        color = colors[int(lbl)].tolist()
        cv2.rectangle(im, (x, y), (x + w, y + h), color, 3)

def save_detection_samples(model, dataset, sAnchors, indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):

    for i in indices:
        im, _, _ = dataset[i]

        pred = model(np.expand_dims(im, 0))

        bboxes = [
                *cell_to_boxes(pred[0].data, S[0], sAnchors=sAnchors[0]),
                *cell_to_boxes(pred[1].data, S[1], sAnchors=sAnchors[1])
             ]
        
        im = np.rollaxis(im.squeeze(), 0, 3)
        im = ((im + 0) * 255).astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        
        plot_boxes(im, bboxes)
        cv2.imwrite(f"{script_dir}/out-{i}.png", im)

def track(sequence, desc_func):
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(),
    )

    with progress:
        tid = progress.add_task(desc_func(), total=len(sequence))

        for value in sequence:
            yield value
            progress.update(tid, description=desc_func(), refresh=True, advance=1)


if __name__ == "__main__":
    sa = get_scaled_anchors()
    print(sa)
