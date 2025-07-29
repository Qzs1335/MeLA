import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    weights = node_attr  # Assuming node_attr is already the weights array
    dist_matrix = np.abs(weights[:,None] - weights)
    heur = 1/(1 + dist_matrix)
    heur *= (weights[:,None] + weights) <= node_constraint
    return heur
    #EVOLVE-END