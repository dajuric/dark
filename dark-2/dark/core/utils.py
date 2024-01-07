import dark.tensor as dt

def reduce_sum(result, target_shape):
    dim_diff = result.ndim - len(target_shape)
    if dim_diff < 0:
        return result.reshape(target_shape)
    
    # sum all leading dimensions
    # except leading dimensions, sum all axis that are equal to 1, count from right
    dim0 = tuple(range(dim_diff))
    dim1 = tuple(i - len(target_shape) for i, value in enumerate(target_shape) if value == 1)
    
    result = result.sum(dim0 + dim1, keepdims=True)
    result = result.squeeze(dim0)
    return result