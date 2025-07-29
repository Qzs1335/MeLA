import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    sizes = node_attr[:,0].reshape(-1,1)  # Assume first column is size
    heuristic = sizes @ sizes.T  # Outer product
    heuristic /= heuristic.max()  # Normalize
    return heuristic + np.eye(n)  # Add self-preference
    #EVOLVE-END