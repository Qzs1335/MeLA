import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    sizes = node_attr[:, 0].reshape(-1, 1)
    capacity = node_constraint[0]
    size_diff = np.abs(sizes - sizes.T)
    heuristic = np.exp(-size_diff/capacity)
    return heuristic * (sizes + sizes.T <= capacity)
    #EVOLVE-END