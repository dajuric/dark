from numpy import *
seterr(over='raise')
    
def unsqueeze(x, axis):
    return expand_dims(x, axis)

def sigmoid(x):
    return 1 / (1 + exp(-x))