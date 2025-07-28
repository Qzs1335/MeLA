import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Ensure node_attr is 2D
    if len(node_attr.shape) == 1:
        node_attr = node_attr.reshape(-1, 1)
    
    n = node_attr.shape[0]
    sizes = node_attr[:,0] if node_attr.shape[1] > 0 else node_attr.flatten()
    heur = np.outer(sizes, sizes)
    return np.exp(heur/np.max(heur))
    #EVOLVE-END