import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Ensure node_attr is treated as 1D array
    weights = np.asarray(node_attr).flatten()
    n = weights.shape[0]
    # Handle zero constraint case to avoid division by zero
    constraint = max(node_constraint, 1e-10)  # small positive value if constraint is zero
    
    h = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i == j:
                h[i,j] = weights[i]/constraint
            else:
                h[i,j] = 1 - abs(weights[i]-weights[j])/constraint
    return h
    #EVOLVE-END