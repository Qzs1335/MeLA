import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    weights = node_attr[:,0]
    dist_matrix = np.abs(weights[:,None] - weights)
    heuristic = 1/(1 + dist_matrix) * (node_constraint[:,None] >= weights)
    return heuristic
    #EVOLVE-END