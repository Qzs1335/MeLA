import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    attr_matrix = np.tile(node_attr, (n,1))
    combined = attr_matrix + attr_matrix.T
    weights = 1/(1 + np.abs(node_constraint - combined))
    return weights/np.max(weights)
    #EVOLVE-END