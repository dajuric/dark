import torch
import torch.nn as nn
from config import *

class YoloLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, S, S, C + (1+4) * 1)
        box_exist = target[..., C].unsqueeze(3)

        #box coordinates loss
        box_predictions = box_exist * predictions[..., (C+1):(C+1+4)]
        box_targets     = box_exist * target[..., (C+1):(C+1+4)]

        wh_preds = box_predictions[..., 2:4].clone()
        box_predictions[..., 2:4] = torch.sign(wh_preds) * torch.sqrt(torch.abs(wh_preds) + 1e-6)
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )
        
        #object loss
        obj_loss = self.mse(
            torch.flatten(box_exist * predictions[..., C:(C+1)], start_dim=1),
            torch.flatten(box_exist * target[..., C:(C+1)],      start_dim=1)
        )

        #no object loss
        no_obj_loss = self.mse(
            torch.flatten((1 - box_exist) * predictions[..., C:(C+1)], start_dim=1),
            torch.flatten((1 - box_exist) * target[..., C:(C+1)],      start_dim=1)
        )

        #class loss
        class_loss = self.mse(
            torch.flatten(box_exist * predictions[..., :C], end_dim=-2),
            torch.flatten(box_exist * target[..., :C],      end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss + 
            obj_loss + 
            self.lambda_noobj * no_obj_loss + 
            class_loss
        )

        return loss