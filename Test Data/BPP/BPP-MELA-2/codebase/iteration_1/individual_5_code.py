import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Ensure node_attr is properly shaped as 2D column vector
    node_attr = np.asarray(node_attr).reshape(-1, 1)
    n = node_attr.shape[0]
    sizes = node_attr[:,0]
    normalized_sizes = sizes / node_constraint
    heuristic = np.outer(normalized_sizes, normalized_sizes)
    return heuristic * (1 + np.eye(n))  # Boost diagonal
    #EVOLVE-END