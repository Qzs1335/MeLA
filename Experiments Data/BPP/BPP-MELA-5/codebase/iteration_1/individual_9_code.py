import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)  # Use len() for 1D arrays
    sizes = node_attr  # Directly use node_attr if it's 1D
    heuristic = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j and sizes[i] + sizes[j] <= node_constraint:
                heuristic[i,j] = 1/(1 + abs(sizes[i] - sizes[j]))
    return heuristic
    #EVOLVE-END