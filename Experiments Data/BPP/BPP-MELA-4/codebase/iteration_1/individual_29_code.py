import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Handle both 1D and 2D node_attr cases
    if len(node_attr.shape) == 1:
        sizes = node_attr.reshape(-1,1)
    else:
        sizes = node_attr[:,0].reshape(-1,1)
    
    capacity = node_constraint[0] if isinstance(node_constraint, (list, np.ndarray)) else node_constraint
    
    size_ratio = np.minimum(sizes, capacity) / capacity
    heuristic = 1 - np.abs(size_ratio - size_ratio.T)
    return heuristic
    #EVOLVE-END