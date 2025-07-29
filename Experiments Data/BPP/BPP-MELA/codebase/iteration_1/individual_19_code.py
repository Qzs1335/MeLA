import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Ensure inputs are numpy arrays
    node_attr = np.asarray(node_attr)
    node_constraint = np.asarray(node_constraint)
    
    # Reshape if 1D array
    if node_attr.ndim == 1:
        node_attr = node_attr.reshape(-1, 1)
    
    # Calculate weights (handle division by zero)
    weights = node_attr[:, 0] / np.maximum(node_constraint, 1e-10)  # Normalized item sizes
    
    # Create probability matrix
    prob_matrix = np.outer(weights, weights)    # Priority for larger items
    prob_matrix /= np.maximum(prob_matrix.sum(axis=1, keepdims=True), 1e-10)  # Normalize rows
    
    return prob_matrix
    #EVOLVE-END