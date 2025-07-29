import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Handle both 1D and 2D node_attr cases
    if len(node_attr.shape) == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:,0] if node_attr.shape[1] > 0 else node_attr[:,0]
    
    n = len(sizes)
    size_ratio = np.outer(sizes, 1/sizes)
    heuristic = 0.7*size_ratio + 0.3*np.random.rand(n,n)
    return (heuristic + heuristic.T)/2  # Symmetric matrix
    #EVOLVE-END