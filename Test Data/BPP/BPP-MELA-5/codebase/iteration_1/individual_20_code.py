import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    if len(node_attr.shape) == 1:
        weights = node_attr  # If 1D, use directly
    else:
        weights = node_attr[:,0]  # If 2D, take first column
    
    dist_matrix = np.abs(weights[:,None] - weights)
    capacity_ratio = np.minimum(weights[:,None], weights) / node_constraint
    return 0.7 * (1/(dist_matrix + 1e-6)) + 0.3 * capacity_ratio  # Added small constant to avoid division by zero
    #EVOLVE-END