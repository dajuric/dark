import torch
import torch.nn as nn
from config import *
from utils import *

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10


    def forward(self, predictions, target, sAnchors):
        obj   = target[..., 0] == 1
        noobj = target[..., 0] == 0

        sAnchors = sAnchors.reshape(1, NUM_ANCHORS, 1, 1, 2)
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        predictions[..., 3:5] = torch.exp(predictions[..., 3:5]) * sAnchors
        target[..., 3:5] = torch.log(target[..., 3:5] / sAnchors + 1e-6)

        #no object loss
        no_obj_loss = self.bce(
            predictions[..., 0:1][noobj],
            target[..., 0:1][noobj]
        )

        #object loss
        ious = iou(predictions[..., 1:5][obj], target[..., 1:5][obj]).detach()
        obj_loss = self.mse(
                        self.sigmoid(predictions[..., 0:1][obj]),
                        ious * target[..., 0:1][obj]
        )

        #box coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        #class loss
        class_loss = self.entropy(
                        predictions[..., 5:][obj], 
                        target[..., 5][obj].long()
        )

        loss = ( 
                 self.lambda_noobj * no_obj_loss +
                 self.lambda_obj   * obj_loss    +
                 self.lambda_box   * box_loss    + 
                 self.lambda_class * class_loss
               )
        return loss

        