import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr) if node_attr.ndim == 1 else node_attr.shape[0]
    sizes = node_attr if node_attr.ndim == 1 else node_attr[:,0]
    h = np.abs(np.subtract.outer(sizes, sizes)) / node_constraint
    h = 1 / (1 + h)  # Inverse relationship
    np.fill_diagonal(h, 0)  # No self-transitions
    h += 0.1 * np.random.rand(*h.shape)  # Exploration with matching dimensions
    return h
    #EVOLVE-END