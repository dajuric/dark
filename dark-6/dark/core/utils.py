import dark.tensor as xp

def reduce_sum(tensor, targetShape):
    assert len(tensor.shape) == len(targetShape)

    result = tensor.copy()
    while True:
        srcShape = result.shape
        
        reduceDim = xp.argmin(xp.array(targetShape) - xp.array(srcShape)).item()
        if srcShape[reduceDim] == targetShape[reduceDim]:
            return result

        result = xp.sum(result, axis=reduceDim, keepdims=True)
  