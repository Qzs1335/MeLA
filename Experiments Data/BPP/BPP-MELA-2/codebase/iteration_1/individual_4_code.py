import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)
    if node_attr.ndim == 1:
        sizes = node_attr  # if 1D, use directly
    else:
        sizes = node_attr[:, 0]  # if 2D, use first column
    
    max_size = np.max(sizes)
    normalized_sizes = sizes / max_size
    heuristic = np.outer(normalized_sizes, normalized_sizes)
    return heuristic + 1e-6  # small constant to avoid zeros
    #EVOLVE-END