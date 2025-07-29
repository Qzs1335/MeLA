import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)
    if node_attr.ndim == 1:
        weights = node_attr
    else:
        weights = node_attr[:,0] if node_attr.shape[1] > 0 else np.zeros(n)
    
    heuristic = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                remaining = node_constraint - weights[i]
                heuristic[i,j] = 1/(1 + abs(weights[j] - remaining))
    return heuristic
    #EVOLVE-END