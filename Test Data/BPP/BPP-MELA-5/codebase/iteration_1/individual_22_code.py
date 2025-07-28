import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    if len(node_attr.shape) == 1:
        sizes = node_attr.reshape(-1,1)
    else:
        sizes = node_attr[:,0].reshape(-1,1) if node_attr.shape[1] > 0 else node_attr.reshape(-1,1)
    heur = 1 - np.abs(sizes - sizes.T)/node_constraint
    np.fill_diagonal(heur, 0)
    return heur
    #EVOLVE-END