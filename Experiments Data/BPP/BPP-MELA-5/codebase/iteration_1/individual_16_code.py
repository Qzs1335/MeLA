import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.asarray(node_attr)
    if node_attr.ndim == 1:
        sizes = node_attr.reshape(-1, 1)
    else:
        sizes = node_attr[:,0]
    n = sizes.shape[0]
    h = np.zeros((n,n))
    for i in range(n):
        remaining = node_constraint - sizes[i]
        for j in range(n):
            if i != j and sizes[j] <= remaining:
                h[i,j] = 1/(1 + sizes[j]/remaining)
    h += 0.01*np.random.rand(n,n)  # exploration factor
    np.fill_diagonal(h, 0)
    return h
    #EVOLVE-END