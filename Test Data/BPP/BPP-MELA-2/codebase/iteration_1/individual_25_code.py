import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)
    sizes = node_attr  # Use directly if 1D, or node_attr[:,0] if 2D
    heuristic = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                heuristic[i,j] = 1/(1 + abs(sizes[i]-sizes[j]))
    np.fill_diagonal(heuristic, 0)
    return heuristic/node_constraint
    #EVOLVE-END