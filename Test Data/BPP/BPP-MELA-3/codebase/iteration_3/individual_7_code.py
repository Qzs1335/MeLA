import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    size_diff = np.abs(node_attr[:, None] - node_attr[None, :])
    constraint_diff = np.maximum(1e-6, node_constraint - (node_attr[:, None] + node_attr[None, :]))
    weights = 1 / (1 + size_diff + constraint_diff)
    return weights / weights.sum(axis=1, keepdims=True)
    #EVOLVE-END