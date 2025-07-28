import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.array(node_attr).reshape(-1,1) if len(node_attr.shape) == 1 else node_attr
    n = node_attr.shape[0]
    sizes = node_attr[:,0]
    heuristic = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                heuristic[i,j] = 1/(1 + abs(sizes[i]-sizes[j]))
    return heuristic
    #EVOLVE-END