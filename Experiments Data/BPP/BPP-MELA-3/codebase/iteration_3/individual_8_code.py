import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    norm_attr = node_attr / node_attr.max()
    attr_diff = np.abs(norm_attr[:, None] - norm_attr[None, :])
    constraint_diff = np.maximum(1e-6, node_constraint - norm_attr.sum())
    weights = np.exp(-attr_diff) / (1 + constraint_diff)
    return weights
    #EVOLVE-END