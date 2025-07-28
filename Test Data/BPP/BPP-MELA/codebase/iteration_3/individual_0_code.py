import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = node_attr if node_attr.ndim == 1 else node_attr[:,0]
    n = len(sizes)
    size_matrix = sizes.reshape(n,1) + sizes
    capacity_diff = size_matrix - node_constraint
    heuristic = 1/(1 + np.exp(0.5*capacity_diff))  # Sigmoid transformation
    np.fill_diagonal(heuristic, 0)
    return heuristic/heuristic.max()
    #EVOLVE-END