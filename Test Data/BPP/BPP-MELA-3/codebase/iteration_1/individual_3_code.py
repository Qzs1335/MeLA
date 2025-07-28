import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    weights = node_attr[:,0]
    inv_weights = 1/(weights + 1e-6)
    heur_matrix = np.outer(inv_weights, inv_weights)
    heur_matrix *= (1 + np.minimum(node_constraint, node_constraint.T))
    return heur_matrix/heur_matrix.max()
    #EVOLVE-END