import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Ensure node_constraint is positive and non-zero
    node_constraint = max(node_constraint, 1e-10)
    
    # Handle both 1D and 2D node_attr cases
    if node_attr.ndim == 1:
        sizes = node_attr / node_constraint
    else:
        sizes = node_attr[:, 0] / node_constraint
        
    heuristic = np.outer(sizes, sizes)
    np.fill_diagonal(heuristic, 0)
    return heuristic
    #EVOLVE-END