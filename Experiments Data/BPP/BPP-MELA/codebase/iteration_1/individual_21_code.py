import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)
    if len(node_attr.shape) == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:,0]
    h = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if sizes[i] + sizes[j] <= node_constraint:
                h[i,j] = 1/(1 + abs(sizes[i] - sizes[j]))
    return h
    #EVOLVE-END