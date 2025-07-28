import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = node_attr if node_attr.ndim == 1 else node_attr[:,0]
    n = len(sizes)
    size_matrix = sizes.reshape(n,1) + sizes
    ratio_matrix = np.minimum(sizes.reshape(n,1)/sizes, sizes/sizes.reshape(n,1))
    heuristic = ratio_matrix * np.exp(-np.abs(size_matrix - node_constraint)/node_constraint)
    np.fill_diagonal(heuristic, 0)
    return heuristic/heuristic.max()
    #EVOLVE-END