import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)  # Works for both 1D and 2D arrays
    if node_attr.ndim == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:,0]
    heuristic = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                heuristic[i,j] = 1/(1 + abs(node_constraint - sizes[i] - sizes[j]))
    return heuristic + np.random.rand(n,n)*0.1
    #EVOLVE-END