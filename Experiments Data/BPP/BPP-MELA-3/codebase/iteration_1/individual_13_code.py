import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    size_diff = np.abs(node_attr[:,None] - node_attr)
    constraint_match = (node_constraint[:,None] + node_constraint) <= 1
    return np.exp(-size_diff) * constraint_match
    #EVOLVE-END