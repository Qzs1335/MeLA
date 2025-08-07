import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    if len(node_attr.shape) == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:, 0] if node_attr.shape[1] > 0 else node_attr[:, 0]
    
    n = len(sizes)
    heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if sizes[i] + sizes[j] <= node_constraint:
                heuristic[i,j] = 1/(1 + abs(sizes[i] - sizes[j]))
    return heuristic
    #EVOLVE-END