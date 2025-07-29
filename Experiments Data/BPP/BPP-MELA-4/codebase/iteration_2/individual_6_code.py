import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = np.asarray(node_attr).flatten() if np.asarray(node_attr).ndim == 1 else np.asarray(node_attr)[:,0]
    n = len(sizes)
    sij = sizes[:,None] + sizes[None,:]
    h = 1/(np.abs(sij - node_constraint) + 1e-6)
    np.fill_diagonal(h, 0)
    return h
    #EVOLVE-END