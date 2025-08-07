import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    attr_sums = node_attr[:, None] + node_attr[None, :] 
    weights = 1/(1e-8 + np.abs(node_constraint - attr_sums))
    return weights
    #EVOLVE-END