import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Convert inputs to numpy arrays if they aren't already
    node_attr = np.asarray(node_attr)
    node_constraint = np.asarray(node_constraint)
    
    n = node_attr.shape[0]
    attr_diff = np.abs(node_attr[:, None] - node_attr[None, :])
    attr_sum = node_attr[:, None] + node_attr[None, :]
    constraint_diff = np.maximum(1e-6, np.abs(node_constraint - attr_sum))
    weights = 1/(1 + 0.5*(attr_diff/node_attr.mean() + constraint_diff/node_constraint.mean()))
    return weights
    #EVOLVE-END