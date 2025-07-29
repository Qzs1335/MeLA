import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Handle both 1D and 2D node_attr cases
    if node_attr.ndim == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:,0]
    
    n = len(sizes)
    heu = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            heu[i,j] = 1 - abs(sizes[i] - sizes[j])/node_constraint
    return np.maximum(heu, 0)
    #EVOLVE-END