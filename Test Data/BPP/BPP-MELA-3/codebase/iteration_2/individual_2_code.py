import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    combined = node_attr[:, None] + node_attr[None, :]
    weights = 1/(1 + np.abs(node_constraint - combined))
    weights = np.clip(weights, 1e-6, 1)  # Numerical stability
    return weights/np.sum(weights)  # Normalization
    #EVOLVE-END