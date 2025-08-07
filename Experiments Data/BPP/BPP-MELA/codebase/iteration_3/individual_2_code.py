import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = node_attr if node_attr.ndim == 1 else node_attr[:,0]
    n = len(sizes)
    capacity_left = np.maximum(node_constraint - sizes.reshape(n,1) - sizes, 0)
    heuristic = 1/(capacity_left + 1e-6)
    np.fill_diagonal(heuristic, 0)
    return heuristic/heuristic.max()
    #EVOLVE-END