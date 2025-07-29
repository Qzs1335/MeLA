import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    attr_matrix = node_attr[:, None] + node_attr[None, :]
    similarity = 1 - np.abs(node_attr[:, None] - node_attr[None, :])/np.max(node_attr)
    weights = similarity/(1e-6 + np.abs(node_constraint - attr_matrix))
    return weights/np.max(weights)
    #EVOLVE-END