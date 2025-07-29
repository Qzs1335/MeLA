import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)
    sizes = node_attr if node_attr.ndim == 1 else node_attr[:, 0]  # Handle both 1D and 2D cases
    heuristic = np.outer(1/(sizes+1e-8), np.ones(n))  # Prefer smaller items
    heuristic *= (node_constraint - sizes.reshape(-1,1))  # Penalize overfilling
    return np.maximum(heuristic, 1e-8)  # Avoid zeros
    #EVOLVE-END