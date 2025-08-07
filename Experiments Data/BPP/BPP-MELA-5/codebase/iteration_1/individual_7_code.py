import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.asarray(node_attr)
    if node_attr.ndim == 1:
        sizes = node_attr.reshape(-1, 1)
    else:
        sizes = node_attr[:, 0].reshape(-1, 1)
    heuristic = np.abs(sizes - sizes.T)  # Difference matrix
    heuristic = 1 / (1 + heuristic)  # Inverse relationship
    np.fill_diagonal(heuristic, 0)  # No self-selection
    return heuristic
    #EVOLVE-END