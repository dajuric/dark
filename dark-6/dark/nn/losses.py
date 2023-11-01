import dark
from .module import Module, Softmax

class BCEWithLogitsLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        predictions = self.sigmoid(predictions)

        lossA = dark.mul(targets, dark.log(predictions))
        lossB = dark.mul(dark.subtract(1, targets), dark.log(dark.subtract(1, predictions)))

        loss  = dark.mean(dark.add(lossA, lossB))
        loss  = dark.neg(loss)
        return loss
    
    def sigmoid(self, x):
        den = dark.add(1, dark.exp(dark.neg(x)))
        return dark.div(1, den)

class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def forward(self, predictions, targets):
        loss = dark.mul(targets, dark.log(self.softmax(predictions)))
        loss = dark.sum(loss, dim=1)
        loss = dark.neg(dark.mean(loss))
        return loss


# import dark
# from .module import Module, Softmax

# class CrossEntropyLoss(Module):
#     def __init__(self):
#         super().__init__()
#         self.softmax = Softmax()

#     def forward(self, predictions, targets):    
#         logSoftmax = self._log_softmax(predictions)

#         loss = dark.mul(targets, logSoftmax)
#         loss = dark.sum(loss, dim=1)
#         loss = dark.neg(dark.mean(loss))
#         return loss

#     def _log_softmax(self, x):
#         result = dark.sum(x, dim=0)
#         result = dark.subtract(x, result)
#         #result = dark.subtract(0, result)
#         return result