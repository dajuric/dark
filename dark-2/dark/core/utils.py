import numpy as np

def reduce_sum(tensor, targetShape):
    assert len(tensor.shape) == len(targetShape)

    result = tensor.copy()
    while True:
        srcShape = result.shape
        
        reduceDim = np.argmin(np.array(targetShape) - np.array(srcShape))
        if srcShape[reduceDim] == targetShape[reduceDim]:
            return result

        result = np.sum(result, axis=reduceDim, keepdims=True)
  