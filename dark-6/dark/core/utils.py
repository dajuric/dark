import dark.tensor as dt

def reduce_sum(tensor, targetShape):
    assert len(tensor.shape) == len(targetShape)

    result = tensor.copy()
    while True:
        srcShape = result.shape
        
        reduceDim = dt.argmin(dt.array(targetShape) - dt.array(srcShape)).item()
        if srcShape[reduceDim] == targetShape[reduceDim]:
            return result

        result = dt.sum(result, axis=reduceDim, keepdims=True)
  