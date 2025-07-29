import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    if node_attr.ndim == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:, 0]
    
    # Ensure node_constraint is positive
    node_constraint = max(node_constraint, 1e-6)
    
    dist = np.abs(sizes[:, None] - sizes)
    heuristic = np.exp(-dist/node_constraint)
    np.fill_diagonal(heuristic, 0)
    return heuristic
    #EVOLVE-END