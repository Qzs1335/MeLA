import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = node_attr if node_attr.ndim == 1 else node_attr[:,0]
    size_diff = np.subtract.outer(sizes, -sizes) - node_constraint
    heuristic = 1/(np.abs(size_diff) + 1e-6)
    np.fill_diagonal(heuristic, 0, wrap=True)
    return heuristic * (1/np.nanmax(heuristic))
    #EVOLVE-END