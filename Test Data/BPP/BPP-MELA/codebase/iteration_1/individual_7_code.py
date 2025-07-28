import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)
    if node_attr.ndim == 1:
        weights = 1/(node_attr + 1e-6)  # Handle 1D array case
    else:
        weights = 1/(node_attr[:,0] + 1e-6)  # Handle 2D array case
    heur = np.outer(weights, weights) * (1 + 0.1*np.random.rand(n,n))
    np.fill_diagonal(heur, 0)  # No self-loops
    return heur
    #EVOLVE-END