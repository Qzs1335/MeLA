import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    
    heuristic = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            remaining_cap = node_constraint
        else:
            remaining_cap = node_constraint - np.sum(node_attr[:i])
        
        for j in range(n):
            if remaining_cap > 0 and node_attr[j] <= remaining_cap:
                heuristic[i,j] = np.exp(-node_attr[j]/remaining_cap)
    return heuristic
    #EVOLVE-END