import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    remaining_capacity = node_constraint - node_attr[:,None]
    valid = (remaining_capacity >= 0).astype(float)
    heuristic = np.exp(-1/(remaining_capacity*valid + 1e-9))
    np.fill_diagonal(heuristic, 0)  # Prevent self-assignment
    return heuristic
    #EVOLVE-END