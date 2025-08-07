import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.asarray(node_attr).reshape(-1,1)  # Ensure 2D column vector
    sizes = node_attr[:,0].reshape(-1,1)  # Extract sizes as column vector
    heuristic = 1 - np.abs(sizes - sizes.T)/node_constraint
    return np.maximum(heuristic, 0.1)
    #EVOLVE-END