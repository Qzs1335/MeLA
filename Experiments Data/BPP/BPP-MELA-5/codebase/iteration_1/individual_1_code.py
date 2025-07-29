import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    if len(node_attr.shape) == 1:
        weights = node_attr
    else:
        weights = node_attr[:, 0]
    weight_matrix = np.minimum.outer(weights, weights)
    capacity_factor = 1 - (weight_matrix / node_constraint)
    return weight_matrix * capacity_factor / np.max(weight_matrix)
    #EVOLVE-END