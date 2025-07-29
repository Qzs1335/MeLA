import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.size
    weights = node_attr if node_attr.ndim == 1 else node_attr[:,0]
    heuristic = np.abs(np.subtract.outer(weights, weights))
    heuristic = 1/(1 + heuristic)  # Similar items get higher values
    capacity_ratio = np.minimum.outer(weights, weights)/node_constraint
    return heuristic * (1 - capacity_ratio)  # Penalize pairs nearing capacity
    #EVOLVE-END