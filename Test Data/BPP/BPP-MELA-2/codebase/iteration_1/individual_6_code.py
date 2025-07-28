import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    if node_attr.ndim == 1:
        weights = node_attr.reshape(-1,1)
    else:
        weights = node_attr[:,0].reshape(-1,1) if node_attr.shape[1] > 0 else node_attr.reshape(-1,1)
    
    weight_diff = np.abs(weights - weights.T)
    heuristic = 1/(1 + weight_diff)
    np.fill_diagonal(heuristic, 0)  # Prevent self-selection
    return heuristic/node_constraint  # Constraint normalization
    #EVOLVE-END