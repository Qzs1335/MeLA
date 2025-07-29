import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    weights = np.array(node_attr)  # Ensure we have a numpy array
    heuristic = np.outer(weights, weights) 
    return heuristic / np.max(heuristic)  # Normalized to [0,1]
    #EVOLVE-END