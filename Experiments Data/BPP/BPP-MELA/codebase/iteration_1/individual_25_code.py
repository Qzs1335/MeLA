import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    constraints = node_constraint.reshape(-1,1) + 1e-10  # avoid division by zero
    heuristic_matrix = 1 / (constraints + constraints.T)
    np.fill_diagonal(heuristic_matrix, 0)  # prevent self-selection
    return heuristic_matrix
    #EVOLVE-END