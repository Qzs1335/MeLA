import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)
    # Handle both 1D and 2D input cases
    if node_attr.ndim == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:,0] if node_attr.shape[1] > 0 else node_attr[:,0:1].flatten()
    
    h = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                denominator = node_constraint - sizes[i]
                if denominator > 0:  # Prevent division by zero
                    h[i,j] = (sizes[i] + sizes[j]) / denominator
    return h
    #EVOLVE-END