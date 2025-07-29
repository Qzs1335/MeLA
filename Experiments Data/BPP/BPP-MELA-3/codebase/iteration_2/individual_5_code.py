import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    attr_sum = node_attr[:, None] + node_attr[None, :]
    diff = np.abs(node_constraint - attr_sum)
    weights = np.exp(-diff/(0.5*node_constraint + 1e-8))
    return weights/np.max(weights)
    #EVOLVE-END