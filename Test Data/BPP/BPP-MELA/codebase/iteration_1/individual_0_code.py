import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)
    if node_attr.ndim == 1:
        weights = node_attr
    else:
        weights = node_attr[:,0]
    capacities = np.full(n, node_constraint)
    ratio_matrix = np.outer(weights, 1/capacities)
    return np.exp(-ratio_matrix)
    #EVOLVE-END