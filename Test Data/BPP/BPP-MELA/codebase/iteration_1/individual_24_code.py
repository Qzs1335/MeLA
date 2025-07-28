import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)
    if node_attr.ndim == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:,0] if node_attr.shape[1] > 0 else node_attr[:,0]
    
    heuristic = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                heuristic[i,j] = (sizes[i] + sizes[j]) / node_constraint
    max_val = np.max(heuristic)
    return heuristic / max_val if max_val > 0 else heuristic
    #EVOLVE-END