import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = node_attr if node_attr.ndim == 1 else node_attr[:,0]
    n = len(sizes)
    size_matrix = np.log1p(sizes.reshape(n,1) + sizes)
    heuristic = np.exp(-np.abs(size_matrix - np.log1p(node_constraint)))
    np.fill_diagonal(heuristic, 0)
    return heuristic/(heuristic.max() + 1e-10)
    #EVOLVE-END