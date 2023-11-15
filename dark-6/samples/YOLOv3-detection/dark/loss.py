import dark
import dark.nn as nn
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
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i
        sAnchors = sAnchors.reshape(1, NUM_ANCHORS, 1, 1, 2)
        
        xym = dt.zeros(predictions); xym[..., 1:3] = True
        whm = dt.zeros(predictions); whm[..., 3:5] = True
        
        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        nom = dt.zeros(predictions); nom[..., 0:1][noobj] = True
        
        no_object_loss = self.bce(
            dark.mask(predictions, nom), 
            target[nom],
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        om = dt.zeros(predictions); om[..., 0:1][obj] = True
        
        box_preds = dark.cat([
                               self.sigmoid(dark.mask(predictions, xym)), 
                               dark.mul(dark.exp(dark.mask(predictions, whm)), sAnchors)
                             ], 
                             dim=-1)
        
        ious = iou(box_preds.data[obj], target[..., 1:5][obj])        
        object_loss = self.mse(self.sigmoid(dark.mask(predictions, om)), 
                               ious * target[om])

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #        
        preds_xywh = dark.cat([
                                 self.sigmoid(dark.mask(predictions, xym)),
                                 dark.mask(predictions, whm)
                              ],
                              dim=-1)
        
        tgts_xywh = dt.concatenate([
                                      target[xym],
                                      dt.log(1e-16 + target[whm] / sAnchors)
                                   ], 
                                   dim=-1)
        
        box_loss = self.mse(preds_xywh, tgts_xywh)

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        cm = dt.zeros(predictions); cm[..., 5:][obj] = True
        
        class_loss = self.entropy(
            dark.mask(predictions, cm), 
            target[..., 5][obj].long()
        )


        loss = dark.cat([
                            dark.mul(self.lambda_box,   box_loss),
                            dark.mul(self.lambda_obj,   object_loss),
                            dark.mul(self.lambda_noobj, no_object_loss),
                            dark.mul(self.lambda_class, class_loss)
                        ])
        loss = dark.sum(loss)
        return loss