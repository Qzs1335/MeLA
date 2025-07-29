import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    weights = node_attr / node_constraint
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)
    similarity = 1 - np.abs(weights - weights.T)
    return similarity / similarity.sum(axis=1, keepdims=True)
    #EVOLVE-END