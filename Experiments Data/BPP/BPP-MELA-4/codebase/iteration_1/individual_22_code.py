import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)
    if node_attr.ndim == 1:
        sizes = node_attr/node_constraint
    else:
        sizes = node_attr[:,0]/node_constraint
    heuristic = np.outer(sizes, 1-sizes) * (1 + np.random.rand(n,n)*0.1)
    np.fill_diagonal(heuristic, 0)
    return heuristic
    #EVOLVE-END