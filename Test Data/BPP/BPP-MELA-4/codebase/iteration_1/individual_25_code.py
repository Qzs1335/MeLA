import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.asarray(node_attr)
    if node_attr.ndim == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:,0] if node_attr.shape[1] > 0 else node_attr[:,0:1].flatten()
    n = len(sizes)
    h = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                h[i,j] = 1/(abs(sizes[i] + sizes[j] - node_constraint) + 1e-6)
    return h
    #EVOLVE-END