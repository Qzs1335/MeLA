import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Ensure node_attr is 2D with shape (n,1)
    node_attr = np.asarray(node_attr)
    if node_attr.ndim == 1:
        node_attr = node_attr.reshape(-1,1)
    
    n = node_attr.shape[0]
    sizes = node_attr[:,0].reshape(-1,1)
    heuristic = 1/(1 + np.abs(sizes - sizes.T))  # Size similarity
    heuristic *= (sizes + sizes.T <= node_constraint).astype(float)  # Capacity check
    np.fill_diagonal(heuristic, 0)  # Prevent self-assignment
    return heuristic
    #EVOLVE-END