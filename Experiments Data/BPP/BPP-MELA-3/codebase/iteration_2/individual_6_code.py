import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    attr_matrix = node_attr.reshape(-1,1) + node_attr
    diff = np.abs(node_constraint - attr_matrix)
    weights = 1 / (1 + np.exp(diff - np.mean(diff)))
    return weights / np.max(weights)
    #EVOLVE-END