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
        target = target.data
        sAnchors = sAnchors.data

        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i
        sAnchors = sAnchors.reshape(1, NUM_ANCHORS, 1, 1, 2)
        
        
        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        nom = dt.zeros_like(predictions.data, dtype=bool); nom[..., 0:1][noobj] = True
        
        no_object_loss = self.bce(
            dark.mask(predictions, nom), 
            target[nom],
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        om = dt.zeros_like(predictions.data, dtype=bool); om[..., 0:1][obj] = True
        
        box_preds = dt.concatenate([
                                    dt.sigmoid(predictions.data[..., 1:3]), 
                                    dt.exp(predictions.data[..., 3:5]) * sAnchors
                                   ], 
                                   axis=-1)
                
        ious = iou(box_preds[obj], target[..., 1:5][obj])        
        object_loss = self.mse(self.sigmoid(dark.mask(predictions, om)), 
                               ious * target[om])

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #  
        xyom = dt.zeros_like(predictions.data, dtype=bool); xyom[..., 1:3][obj] = True
        whom = dt.zeros_like(predictions.data, dtype=bool); whom[..., 3:5][obj] = True
        
        xy_loss = self.mse(
                        self.sigmoid(dark.mask(predictions, xyom)),
                        dt.expand_dims(target[xyom], 0)
                          )
        
        wh_loss = self.mse(
                        dark.mask(predictions, whom),
                        dt.log(1e-16 + target[..., 3:5] / sAnchors)[obj].reshape(-1)
                          )
        
        box_loss = dark.add(xy_loss, wh_loss)

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        cm = dt.zeros_like(predictions.data, dtype=bool); cm[..., 5:][obj] = True
        classes = target[..., 5][obj].astype(dt.int32)
        classes_onehot = dt.eye(C)[classes]
        
        class_loss = self.entropy(
            dark.reshape(dark.mask(predictions, cm), (-1, C)), 
            classes_onehot
        )


        # FINAL LOSS
        loss = dark.cat([
                            dark.mul(self.lambda_box,   box_loss),
                            dark.mul(self.lambda_obj,   object_loss),
                            dark.mul(self.lambda_noobj, no_object_loss),
                            dark.mul(self.lambda_class, class_loss)
                        ])
        loss = dark.sum(loss)
        return loss