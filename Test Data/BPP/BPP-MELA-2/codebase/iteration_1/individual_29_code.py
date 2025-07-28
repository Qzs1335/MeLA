import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    weights = node_attr / node_constraint
    heuristic = np.outer(weights, weights)
    return np.exp(-heuristic)  # Inverse relationship
    #EVOLVE-END