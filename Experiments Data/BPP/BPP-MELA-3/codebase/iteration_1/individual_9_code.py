import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Ensure sizes is a 1D array
    sizes = np.asarray(node_attr).flatten()
    n = len(sizes)
    # Calculate pairwise differences correctly
    size_diff = np.abs(sizes.reshape(-1,1) - sizes.reshape(1,-1))
    norm_diff = size_diff / node_constraint
    heuristic = 1 / (1 + norm_diff)
    np.fill_diagonal(heuristic, 0)
    return heuristic
    #EVOLVE-END