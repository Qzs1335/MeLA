import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    assert node_attr.size > 0, "Empty input"
    sizes = node_attr if node_attr.ndim == 1 else node_attr[:,0]
    n = sizes.size
    size_matrix = sizes[:,None] + sizes[None,:]
    h = 1/(np.abs(size_matrix - node_constraint) + 1e-6)
    np.fill_diagonal(h, 0)
    return h
    #EVOLVE-END       
    return h