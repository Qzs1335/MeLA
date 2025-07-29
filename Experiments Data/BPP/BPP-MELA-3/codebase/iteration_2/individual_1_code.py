import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    combined = node_attr[:, None] + node_attr[None, :]
    weights = 1/(1e-8 + np.abs(node_constraint - combined))
    return weights
    #EVOLVE-END