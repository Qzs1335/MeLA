import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    weights = node_attr if len(node_attr.shape) == 1 else node_attr[:,0]
    heuristic = np.abs(np.subtract.outer(weights, weights)) / node_constraint
    return 1 / (1 + heuristic)
    #EVOLVE-END