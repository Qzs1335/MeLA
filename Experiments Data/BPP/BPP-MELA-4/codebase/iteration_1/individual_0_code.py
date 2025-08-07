import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    weights = 1/(1 + node_constraint)  # Inverse weighting
    heuristic = np.outer(weights, weights)  # Pairwise combinations
    heuristic += 0.1*np.random.rand(n,n)  # Exploration factor
    return heuristic/heuristic.sum()  # Normalization
    #EVOLVE-END