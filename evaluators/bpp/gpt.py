import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    n = node_attr.shape[0]
    #EVOLVE-START
    # Handle division by zero in normalization
    max_vals = np.max(node_attr, axis=0)
    max_vals[max_vals == 0] = 1  # Avoid division by zero
    norm_attr = node_attr / max_vals
    
    # Calculate pairwise distances efficiently
    dist_matrix = np.sqrt(np.sum((norm_attr[:, np.newaxis, :] - norm_attr[np.newaxis, :, :])**2, axis=-1))
    
    # Return similarity matrix (avoid division by zero)
    dist_matrix[dist_matrix == 0] = 1e-10  # Small value to avoid division by zero
    return 1 / (1 + dist_matrix)
    #EVOLVE-END
