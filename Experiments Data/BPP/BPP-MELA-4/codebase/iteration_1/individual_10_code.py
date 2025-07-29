import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            remaining = node_constraint - node_attr[i]
            heuristic[i,j] = remaining if remaining >= 0 else 0
    return heuristic/node_constraint  # Normalize
    #EVOLVE-END