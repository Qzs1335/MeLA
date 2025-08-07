import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                remaining_capacity = node_constraint - node_attr[i]
                heuristic[i,j] = 1.0/(1 + abs(remaining_capacity - node_attr[j]))
    return heuristic
    #EVOLVE-END