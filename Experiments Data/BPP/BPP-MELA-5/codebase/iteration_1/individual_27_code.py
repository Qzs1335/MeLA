import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    weights = node_attr  # assuming node_attr is already the weights array
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)  # convert to 2D if needed
    h = np.abs(weights - weights.T)  # pairwise absolute differences
    h = 1/(1+h)  # inverse distance
    np.fill_diagonal(h, 0)  # avoid self-selection
    return h
    #EVOLVE-END