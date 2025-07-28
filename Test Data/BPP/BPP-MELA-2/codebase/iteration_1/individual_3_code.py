import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    if node_attr.ndim == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:,0]
    n = len(sizes)
    heuristic = np.outer(sizes, sizes) / (node_constraint**2)
    heuristic += 0.1 * np.random.rand(n,n)  # exploration factor
    np.fill_diagonal(heuristic, 0)  # avoid self-packing
    return heuristic
    #EVOLVE-END