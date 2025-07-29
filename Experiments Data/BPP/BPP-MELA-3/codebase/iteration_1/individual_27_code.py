import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)
    if node_attr.ndim == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:,0] if node_attr.shape[1] > 0 else node_attr[:,0]
    heuristic = 1/(1 + np.abs(sizes - sizes[:, np.newaxis]))
    heuristic += 0.01 * np.random.rand(n,n)
    np.fill_diagonal(heuristic, 0)
    return heuristic
    #EVOLVE-END