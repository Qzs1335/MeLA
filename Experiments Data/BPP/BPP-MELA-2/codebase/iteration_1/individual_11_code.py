import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)
    weights = node_attr  # directly use the 1D array
    inv_weights = 1/(weights + 1e-6)
    heur = np.outer(inv_weights, inv_weights)
    heur = heur * (0.9 + 0.1*np.random.rand(n,n))
    return heur/heur.max()
    #EVOLVE-END