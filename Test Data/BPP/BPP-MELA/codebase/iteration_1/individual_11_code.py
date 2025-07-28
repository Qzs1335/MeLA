import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Ensure inputs are numpy arrays and have correct shapes
    node_attr = np.asarray(node_attr)
    node_constraint = np.asarray(node_constraint)
    
    # Get number of nodes
    n = node_attr.shape[0]
    
    # Handle 1D case (if node_attr is just sizes)
    if node_attr.ndim == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:,0]
    
    # Calculate size difference matrix
    size_matrix = np.abs(sizes[:,None] - sizes)
    
    # Calculate capacity matrix with broadcasting
    capacity_matrix = np.minimum(node_constraint - sizes[:,None], sizes)
    
    # Compute heuristic values
    heur = 1/(1 + size_matrix) * capacity_matrix
    
    # Normalize and return
    return heur/heur.sum(axis=1, keepdims=True)
    #EVOLVE-END