import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    if node_attr.ndim == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:,0]
    
    # Add small epsilon to avoid division by zero
    constraint = node_constraint + 1e-10
    heuristic = np.outer(sizes, sizes) / constraint
    np.fill_diagonal(heuristic, 0)  # Avoid self-pairing
    return heuristic
    #EVOLVE-END