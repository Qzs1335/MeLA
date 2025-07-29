import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = np.asarray(node_attr).reshape(-1, 1)  # Ensure 2D column vector
    size_diff = np.abs(sizes - sizes.T)
    # Add small epsilon to avoid division by zero
    capacity_ratio = sizes / (node_constraint + 1e-10)
    heuristic = (1/(1+size_diff)) * (1-capacity_ratio)
    return heuristic
    #EVOLVE-END