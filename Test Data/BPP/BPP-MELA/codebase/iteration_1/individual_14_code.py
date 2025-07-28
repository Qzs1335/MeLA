import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    if node_attr.ndim == 1:
        weights = node_attr/node_constraint
    else:
        weights = node_attr[:,0]/node_constraint
    heuristic = np.outer(weights, 1-weights)
    np.fill_diagonal(heuristic, 0)
    return heuristic + 1e-6  # Small constant for exploration
    #EVOLVE-END