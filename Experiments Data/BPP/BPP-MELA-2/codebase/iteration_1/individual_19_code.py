import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Handle 1D or 2D input arrays
    weights = (node_attr[:,0] if node_attr.ndim > 1 else node_attr) / node_constraint
    heuristic = np.outer(weights, weights)
    return heuristic / heuristic.max()
    #EVOLVE-END