import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.asarray(node_attr)
    if node_attr.ndim == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:,0]
    
    n = sizes.shape[0]
    if n == 0:
        return np.zeros((0,0))
    
    h = np.abs(np.subtract.outer(sizes, sizes)) 
    h = 1/(1 + h)  # Inverse difference heuristic
    np.fill_diagonal(h, 0)  # Prevent self-selection
    h += np.random.rand(n,n)*0.1  # Add exploration
    return h
    #EVOLVE-END