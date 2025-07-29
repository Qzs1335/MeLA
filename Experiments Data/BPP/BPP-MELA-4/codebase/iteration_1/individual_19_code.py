import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    h = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                h[i,j] = 1 - abs(node_attr[i]-node_attr[j])/node_constraint
    return np.maximum(h, 0)
    #EVOLVE-END