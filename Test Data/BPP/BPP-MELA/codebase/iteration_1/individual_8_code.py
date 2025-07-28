import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    if node_attr.ndim == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:, 0]
    n = sizes.shape[0]
    size_ratio = np.outer(sizes, 1/sizes)
    np.fill_diagonal(size_ratio, 0)
    heuristic = (size_ratio + np.random.rand(n,n)*0.1).clip(0,1)
    return heuristic
    #EVOLVE-END