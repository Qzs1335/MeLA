import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)
    sizes = node_attr if node_attr.ndim == 1 else node_attr[:,0]
    sizes = sizes.reshape(-1,1)
    h = 1/(np.abs(sizes + sizes.T - node_constraint) + 1e-6)
    np.fill_diagonal(h, 0)
    return h
    #EVOLVE-END