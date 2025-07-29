import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                combined = node_attr[i] + node_attr[j]
                heuristic[i,j] = 1 - (combined - node_constraint)/node_constraint if combined <= node_constraint else 0
    np.fill_diagonal(heuristic, 0)
    return heuristic
    #EVOLVE-END