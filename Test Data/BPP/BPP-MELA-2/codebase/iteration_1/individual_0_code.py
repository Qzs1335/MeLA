import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Ensure node_attr is 2D (n_samples, n_features)
    if len(node_attr.shape) == 1:
        weights = node_attr.reshape(-1, 1)
    else:
        weights = node_attr[:, 0].reshape(-1, 1)  # First column contains item sizes
    
    prob_matrix = np.outer(weights, weights) / (node_constraint ** 2)
    np.fill_diagonal(prob_matrix, 0)  # Prevent self-selection
    return prob_matrix
    #EVOLVE-END