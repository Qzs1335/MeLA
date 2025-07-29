import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    if len(node_attr.shape) == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:, 0]
    n = sizes.shape[0]
    heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                heuristic[i,j] = 1/(abs(sizes[i]-sizes[j]) + 1e-6)
    np.fill_diagonal(heuristic, 0)
    return heuristic
    #EVOLVE-END