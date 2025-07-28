import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    dist_matrix = np.abs(node_attr[:,None] - node_attr)
    heuristic = 1/(1 + dist_matrix)
    np.fill_diagonal(heuristic, 0)
    heuristic *= (node_constraint - node_attr[:,None] - node_attr > 0)
    return heuristic
    #EVOLVE-END