import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    weights = node_attr[:,0]
    normalized_weights = weights / np.max(weights)
    n = len(weights)
    heuristic = np.abs(normalized_weights[:,None] - normalized_weights[None,:])
    return 1 / (1 + heuristic)
    #EVOLVE-END