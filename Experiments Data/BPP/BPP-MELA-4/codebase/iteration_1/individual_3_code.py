import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    dist = np.abs(node_attr[:, None] - node_attr) / node_constraint
    heur = 1/(1 + dist) + 0.1*np.random.rand(n,n)
    return heur/heur.max()
    #EVOLVE-END