import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = np.asarray(node_attr).flatten()
    n = len(sizes)
    i,j = np.indices((n,n))
    mask = i != j
    combined = sizes[i] + sizes[j] - node_constraint
    h = np.zeros((n,n))
    h[mask] = 1/(np.abs(combined[mask]) + 1e-6)
    return h
    #EVOLVE-END