import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)
    weights = node_attr  # Treat node_attr directly as weights
    dist_matrix = np.abs(weights[:, None] - weights[None, :])
    heuristic = 1/(1 + dist_matrix)  # Inverse distance
    heuristic += 0.1*np.random.rand(n, n)  # Exploration factor
    np.fill_diagonal(heuristic, 0)  # No self-pairing
    return heuristic
    #EVOLVE-END