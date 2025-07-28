import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.asarray(node_attr)
    sizes = node_attr.flatten() if node_attr.ndim == 1 else node_attr[:,0]
    n = len(sizes)
    size_sum = sizes[:,None] + sizes[None,:]
    h = 1/(np.abs(size_sum - node_constraint) + 1e-6)
    np.fill_diagonal(h, 0)
    return h
    #EVOLVE-END