import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    if node_attr.ndim == 1:
        weights = node_attr
    else:
        weights = node_attr[:,0] if node_attr.shape[1] > 0 else node_attr[:,0]
    
    heuristic = np.outer(weights, 1/weights)
    np.fill_diagonal(heuristic, 0)
    
    # Avoid division by zero
    denominator = max(node_constraint, 1e-10) * 2
    return heuristic / denominator
    #EVOLVE-END