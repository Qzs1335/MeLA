import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)
    sizes = node_attr.reshape(-1, 1)  # Properly reshape to column vector
    size_diff = np.abs(sizes - sizes.T)
    heuristic = 1.0 / (1.0 + size_diff)
    np.fill_diagonal(heuristic, 0)
    return heuristic
    #EVOLVE-END