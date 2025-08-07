import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = node_attr if node_attr.ndim == 1 else node_attr[:,0]
    n = len(sizes)
    size_ratio = np.minimum(sizes[:,None], sizes) / np.maximum(sizes[:,None], sizes)
    size_matrix = sizes.reshape(n,1) + sizes
    heuristic = (1/(np.abs(size_matrix - node_constraint) + 1e-6)) * (0.5 + 0.5*size_ratio)
    np.fill_diagonal(heuristic, 0)
    return heuristic/heuristic.max()
    #EVOLVE-END