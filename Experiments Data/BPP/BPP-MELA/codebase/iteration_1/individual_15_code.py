import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.atleast_1d(node_attr)
    n = len(node_attr)
    sizes = node_attr if node_attr.ndim == 1 else node_attr[:,0]
    heuristic = np.zeros((n,n))
    for i in range(n):
        remaining = node_constraint - sizes[i]
        heuristic[i] = 1/(1 + np.abs(sizes - remaining))
    heuristic = heuristic/heuristic.sum(axis=1, keepdims=True)
    return heuristic
    #EVOLVE-END