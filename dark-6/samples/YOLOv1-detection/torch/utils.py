import cv2
from config import *

def boxes_from_prediction(predictions, threshold = 0.5):
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, C + (1+4) * 1)
    predBB = predictions[..., (C+1):(C+1+4)]
    predScore = predictions[..., C]
    predLabels = torch.argmax(predictions[..., :C], dim=-1)

    bboxes = []
    for batchIdx in range(batch_size):
        for cellRow in range(S):
            for cellCol in range(S):
                p = predScore[batchIdx, cellRow, cellCol].item()
                if p < threshold:
                    continue

                cellX, cellY, cellW, cellH = predBB[batchIdx, cellRow, cellCol, ...].numpy()
                x = 1 / S * (cellX + cellCol)
                y = 1 / S * (cellY + cellRow)
                w = 1 / S * (cellW)
                h = 1 / S * (cellH)

                lbl = predLabels[batchIdx, cellRow, cellCol].item()
                bb = [x, y, w, h, p, lbl]
                bboxes.append(bb)

    return bboxes


def plot_boxes(image, boxes):
    colors = (torch.rand((C, 3)) * 255).int()
    imH, imW, _ = image.shape

    for bb in boxes:
        bxC, byC, bw, bh, p, lbl = bb
        x = int((bxC - bw / 2) * imW)
        y = int((byC - bh / 2) * imH)
        w = int(bw * imW)
        h = int(bh * imH)

        color = colors[lbl].numpy().tolist()
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)