import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    dists = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            dists[i,j] = np.linalg.norm(node_attr[i]-node_attr[j])
    return 1/(1+dists)
    #EVOLVE-END