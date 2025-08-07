import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.asarray(node_attr)
    sizes = node_attr if node_attr.ndim == 1 else node_attr[:,0] if node_attr.shape[1] > 0 else node_attr[:,0:1].flatten()
    n = len(sizes)
    size_matrix = sizes.reshape(-1,1) + sizes
    h = 1/(np.abs(size_matrix - node_constraint) + 1e-6)
    np.fill_diagonal(h, 0)
    return h
    #EVOLVE-END