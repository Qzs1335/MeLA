import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = node_attr if node_attr.ndim == 1 else node_attr[:, 0]
    n = len(sizes)
    size_matrix = sizes[:, None] + sizes[None, :]
    np.fill_diagonal(size_matrix, 0)
    heuristic = 1 / (np.abs(size_matrix - node_constraint) + 1e-6)
    return heuristic / heuristic.max()
    #EVOLVE-END