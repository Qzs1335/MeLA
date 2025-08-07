import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.asarray(node_attr)
    sizes = node_attr[:,0] if node_attr.ndim > 1 else node_attr
    n = len(sizes)
    s_mat = sizes[:,None] + sizes[None,:]
    h = 1/(np.abs(s_mat - node_constraint) + 1e-6)
    np.fill_diagonal(h, 0)
    return h
    #EVOLVE-END