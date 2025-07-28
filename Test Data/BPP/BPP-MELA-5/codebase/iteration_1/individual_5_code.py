import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    if len(node_attr.shape) == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:, 0]
    heuristic = np.abs(np.subtract.outer(sizes, sizes))
    heuristic = 1 / (heuristic + 1e-6)  # Avoid division by zero
    np.fill_diagonal(heuristic, 0)  # Prevent self-selection
    return heuristic
    #EVOLVE-END