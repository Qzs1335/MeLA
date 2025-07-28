import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    if len(node_attr.shape) == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:,0]
        
    heuristic = np.abs(sizes[:,None] - sizes[None,:]) + 1e-6  # Avoid division by zero
    heuristic = 1 / heuristic  # Prefer similar sizes
    np.fill_diagonal(heuristic, 0)  # No self-connections
    return heuristic / heuristic.max()  # Normalize
    #EVOLVE-END