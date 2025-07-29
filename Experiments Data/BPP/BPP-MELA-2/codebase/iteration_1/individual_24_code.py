import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Handle both 1D and 2D input cases
    if node_attr.ndim == 1:
        weights = node_attr.reshape(-1, 1)
    else:
        weights = node_attr[:, 0].reshape(-1, 1)
    
    # Calculate pairwise differences and apply transformation
    h = np.abs(weights - weights.T)
    h = 1/(1+h)  # Prefer similar weights
    
    # Apply constraint (sum of weights must be <= node_constraint)
    h *= (weights + weights.T <= node_constraint)
    
    np.fill_diagonal(h, 0)  # Avoid self-selection
    return h
    #EVOLVE-END