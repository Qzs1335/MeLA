import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = node_attr  # directly use the 1D array
    n = len(sizes)
    heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                heuristic[i,j] = sizes[i]/(node_constraint - sizes[j] + 1e-9)
    return heuristic
    #EVOLVE-END