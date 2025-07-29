import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    if len(node_attr.shape) == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:, 0] if node_attr.shape[1] > 0 else node_attr[:, 0:1].flatten()
    
    heuristic = np.outer(sizes, sizes) / node_constraint
    np.fill_diagonal(heuristic, 0)
    return heuristic
    #EVOLVE-END