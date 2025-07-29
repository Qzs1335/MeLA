import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    norm_attr = node_attr / node_attr.max()
    attr_sum = norm_attr[:, None] + norm_attr[None, :]
    constraint_diff = np.maximum(1e-6, node_constraint - attr_sum)
    weights = np.exp(-constraint_diff * np.log(2)/0.1)
    return weights
    #EVOLVE-END