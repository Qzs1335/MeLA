import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Handle both 1D and 2D input cases
    if len(node_attr.shape) == 1:
        weights = node_attr
    else:
        weights = node_attr[:,0]
    
    capacity = node_constraint
    # Create inverse weight-distance matrix using broadcasting
    dist_matrix = np.abs(weights[:,None] - weights)
    heuristic = (capacity - dist_matrix) / capacity
    np.fill_diagonal(heuristic, 0)  # Avoid self-selection
    return heuristic
    #EVOLVE-END