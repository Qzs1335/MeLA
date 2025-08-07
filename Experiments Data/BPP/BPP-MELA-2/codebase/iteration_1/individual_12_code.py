import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    weights = node_attr  # directly use the 1D array
    heuristic = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                heuristic[i,j] = 1/(1 + abs(weights[i]-weights[j])) 
    return heuristic * (node_constraint - weights.reshape(-1,1))
    #EVOLVE-END