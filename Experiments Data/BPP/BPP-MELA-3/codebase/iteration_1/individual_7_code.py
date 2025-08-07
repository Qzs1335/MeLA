import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    if node_attr.ndim == 1:
        sizes = node_attr.reshape(-1, 1)
    else:
        sizes = node_attr[:, 0].reshape(-1, 1)
    normalized_sizes = sizes / node_constraint
    heuristic = 1.0 / (1.0 + np.exp(-10*(normalized_sizes - 0.5)))
    heuristic_matrix = heuristic @ heuristic.T
    return heuristic_matrix
    #EVOLVE-END