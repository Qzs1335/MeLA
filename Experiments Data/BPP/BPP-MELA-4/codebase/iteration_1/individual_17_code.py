import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                remaining = node_constraint - node_attr[i] - node_attr[j]
                heuristic[i,j] = 1 - abs(remaining)/node_constraint
    return np.maximum(heuristic, 1e-6)  # Avoid zeros
    #EVOLVE-END