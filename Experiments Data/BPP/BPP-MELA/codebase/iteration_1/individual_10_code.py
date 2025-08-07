import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)
    if node_attr.ndim == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:, 0]
    heuristic = np.outer(sizes, 1/sizes)  # Favor similar sizes
    np.fill_diagonal(heuristic, 0)  # Prevent self-selection
    return heuristic / heuristic.sum(axis=1, keepdims=True)  # Normalize
    #EVOLVE-END