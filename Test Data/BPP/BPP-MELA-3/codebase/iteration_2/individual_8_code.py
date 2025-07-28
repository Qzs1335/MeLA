import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    attr_sum = node_attr[:, None] + node_attr[None, :]
    constraint_diff = np.maximum(1e-6, abs(node_constraint - attr_sum))
    weights = 1 / (1 + constraint_diff * node_attr.mean())
    return weights
    #EVOLVE-END