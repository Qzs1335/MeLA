import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)
    h = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                h[i,j] = 1 - abs(node_attr[i] + node_attr[j] - node_constraint)/node_constraint
    return h + 0.1*np.random.rand(n,n)
    #EVOLVE-END