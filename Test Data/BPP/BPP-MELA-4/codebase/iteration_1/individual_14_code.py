import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.asarray(node_attr).reshape(-1, 1)  # Ensure 2D array
    sizes = node_attr[:,0].reshape(-1,1)
    ratios = np.minimum(sizes, node_constraint)/np.maximum(sizes, node_constraint)
    remaining = node_constraint - sizes.T
    return ratios * (remaining > 0) * (1 - remaining/node_constraint)
    #EVOLVE-END