import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.asarray(node_attr)
    node_constraint = np.asarray(node_constraint)
    
    n = node_attr.shape[0]
    attr_norm = node_attr / node_attr.max()
    scaled_sum = attr_norm[:, None] + attr_norm[None, :]
    constraint_std = node_constraint.std() if node_constraint.size > 1 else 1.0
    constraint_sat = np.exp(-abs(node_constraint - scaled_sum) / constraint_std)
    weights = constraint_sat * (1 + np.tanh(scaled_sum))
    return weights / weights.max()
    #EVOLVE-END