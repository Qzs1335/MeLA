import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    if len(node_attr.shape) == 1:
        weights = node_attr
    else:
        weights = node_attr[:, 0] if node_attr.shape[1] > 0 else node_attr[:, 0]
    
    weights = np.asarray(weights)
    diff_matrix = np.abs(weights.reshape(-1, 1) - weights.reshape(1, -1))
    heuristic = 1 / (diff_matrix + 1e-6)
    np.fill_diagonal(heuristic, 0)
    sums = heuristic.sum(axis=1, keepdims=True)
    return heuristic / (sums + (sums == 0))  # avoid division by zero
    #EVOLVE-END