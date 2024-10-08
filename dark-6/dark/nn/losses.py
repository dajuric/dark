import dark
from .module import Module
from .modules import Sigmoid, Softmax

class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        loss = dark.subtract(predictions, targets)
        loss = dark.pow(loss, 2)
        loss = dark.mean(loss)

        return loss

class BCEWithLogitsLoss(Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = Sigmoid()

    def forward(self, predictions, targets):
        EPS = 1e-10
        predictions = self.sigmoid(predictions)

        lossA = dark.mul(targets, dark.log(dark.add(predictions, EPS)))
        lossB = dark.mul(dark.subtract(1, targets), dark.log(dark.add(dark.subtract(1, predictions), EPS)))
        
        loss  = dark.mean(dark.add(lossA, lossB))
        loss  = dark.neg(loss)
        return loss
    
class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def forward(self, predictions, targets):
        loss = dark.mul(targets, dark.log(self.softmax(predictions)))
        loss = dark.sum(loss, dim=1)
        loss = dark.neg(dark.mean(loss))
        return loss