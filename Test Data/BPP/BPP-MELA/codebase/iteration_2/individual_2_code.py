import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = node_attr if node_attr.ndim == 1 else node_attr[:,0]
    n = len(sizes)
    size_matrix = sizes[:,None] + sizes[None,:]
    constraint_diff = np.abs(size_matrix - node_constraint)
    heuristic = 1/(constraint_diff + 1e-6) * (sizes[:,None]/sizes[None,:])
    np.fill_diagonal(heuristic, 0)
    return heuristic/heuristic.max()
    #EVOLVE-END