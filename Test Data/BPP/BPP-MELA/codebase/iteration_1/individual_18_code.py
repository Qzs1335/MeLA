import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    sizes = node_attr[:,0]  # Assume first column contains item sizes
    heuristic = 1/(sizes[:,None] + 1e-10)  # Inverse size relationship
    heuristic += np.random.rand(n,n)*0.1  # Add exploration factor
    return heuristic/node_constraint  # Normalize by bin capacity
    #EVOLVE-END