import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = node_attr if node_attr.ndim == 1 else node_attr[:,0]
    n = len(sizes)
    size_ratios = np.minimum(sizes.reshape(n,1)/sizes, sizes/sizes.reshape(n,1))
    size_matrix = sizes.reshape(n,1) + sizes
    heuristic = size_ratios * np.exp(-0.5*np.abs(size_matrix - node_constraint))
    np.fill_diagonal(heuristic, 0)
    return heuristic/heuristic.max()
    #EVOLVE-END