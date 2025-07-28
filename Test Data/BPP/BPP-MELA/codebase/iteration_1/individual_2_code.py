import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Ensure node_attr is 2D array
    node_attr = np.array(node_attr)
    if node_attr.ndim == 1:
        node_attr = node_attr.reshape(-1, 1)
    
    n = node_attr.shape[0]
    sizes = node_attr[:,0] if node_attr.shape[1] > 0 else node_attr.ravel()
    sizes = sizes.reshape(-1,1)
    compat_matrix = 1 - np.abs(sizes - sizes.T)/node_constraint
    np.fill_diagonal(compat_matrix, 0)
    return np.clip(compat_matrix, 0, 1)
    #EVOLVE-END