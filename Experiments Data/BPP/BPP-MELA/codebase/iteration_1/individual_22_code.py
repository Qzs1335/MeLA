import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Ensure node_attr is 2D array (n_samples, n_features)
    node_attr = np.array(node_attr).reshape(-1, 1) if len(node_attr.shape) == 1 else node_attr
    
    sizes = node_attr[:,0] if node_attr.shape[1] > 0 else node_attr.flatten()
    norm_sizes = sizes/node_constraint
    
    # Calculate heuristic matrix using broadcasting
    heuristic = np.exp(-5*norm_sizes[:, np.newaxis]) * np.exp(-5*norm_sizes)
    np.fill_diagonal(heuristic, 0)
    return heuristic
    #EVOLVE-END