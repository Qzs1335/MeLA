import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    attr_sim = 1 - np.abs(node_attr[:,None] - node_attr) / np.max(node_attr)
    combined = node_attr[:,None] + node_attr
    constraint_fit = 1/(1 + np.abs(node_constraint - combined))
    weights = 0.5*attr_sim + 0.5*constraint_fit
    return weights/np.max(weights)
    #EVOLVE-END