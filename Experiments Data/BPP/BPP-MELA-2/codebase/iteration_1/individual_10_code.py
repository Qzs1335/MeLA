import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)
    if node_attr.ndim == 2:
        weights = node_attr[:, 0]
    else:
        weights = node_attr
    heur = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            heur[i,j] = 1/(1 + abs(weights[i] - weights[j]))
    return heur
    #EVOLVE-END