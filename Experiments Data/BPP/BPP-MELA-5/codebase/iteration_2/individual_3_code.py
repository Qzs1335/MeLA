import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = node_attr.ravel()
    n = len(sizes)
    sum_matrix = sizes[:, None] + sizes
    h = 1/(np.abs(sum_matrix - node_constraint) + 1e-6)
    np.fill_diagonal(h, 0)
    return h
    #EVOLVE-END