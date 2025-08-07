import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    attr_sum = node_attr[:, None] + node_attr
    weights = 1/(1e-8 + np.abs(node_constraint - attr_sum))
    return weights
    #EVOLVE-END