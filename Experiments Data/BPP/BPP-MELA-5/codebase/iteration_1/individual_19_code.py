import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Ensure node_attr is at least 2D
    if node_attr.ndim == 1:
        node_attr = node_attr.reshape(-1, 1)
    
    # Handle both scalar and array constraints
    if np.isscalar(node_constraint):
        norm_weights = node_attr[:, 0] / node_constraint
    else:
        norm_weights = node_attr[:, 0] / node_constraint[:, 0]
    
    heuristic = np.abs(norm_weights[:, None] - norm_weights)
    return 1 / (heuristic + 1e-6)  # Inverse with small epsilon
    #EVOLVE-END