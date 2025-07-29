import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Ensure node_attr is 2D (Nx1)
    if len(node_attr.shape) == 1:
        node_attr = node_attr.reshape(-1, 1)
    
    n = node_attr.shape[0]
    sizes = node_attr[:,0]
    heuristic = np.outer(sizes, 1/sizes)  # Larger items prefer smaller ones
    heuristic /= np.max(heuristic)       # Normalize to [0,1] range
    np.fill_diagonal(heuristic, 0)       # Avoid self-pairing
    return heuristic
    #EVOLVE-END