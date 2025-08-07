import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    attr_sum = node_attr[:, None] + node_attr[None, :]
    attr_sim = 1 - abs(node_attr[:, None] - node_attr[None, :])/(node_attr.max()+1e-6)
    constraint_diff = np.maximum(1e-6, abs(node_constraint - attr_sum))
    weights = attr_sim / (1 + constraint_diff * node_attr.mean())
    return weights
    #EVOLVE-END