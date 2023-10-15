from dark.tensor import *

def reduce_sum(tensor, targetShape):
    assert len(tensor.shape) == len(targetShape)

    result = tensor.copy()
    while True:
        srcShape = result.shape
        
        reduceDim = cp.argmin(cp.array(targetShape) - cp.array(srcShape)).item()
        if srcShape[reduceDim] == targetShape[reduceDim]:
            return result

        result = cp.sum(result, axis=reduceDim, keepdims=True)
  