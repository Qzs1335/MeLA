import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)
    sizes = np.asarray(node_attr).reshape(-1,1)  # Ensure 2D column vector
    size_diff = np.abs(sizes - sizes.T)
    affinity = 1/(1 + size_diff)
    np.fill_diagonal(affinity, 0)
    constraint_mask = (sizes + sizes.T <= node_constraint).astype(float)
    return affinity * constraint_mask
    #EVOLVE-END