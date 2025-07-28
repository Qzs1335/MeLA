import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    sizes = node_attr[:,0].reshape(-1,1)
    remaining = node_constraint - np.cumsum(sizes)
    heur = np.exp(sizes / remaining.clip(min=1e-6))
    return heur / heur.sum(axis=1, keepdims=True)
    #EVOLVE-END