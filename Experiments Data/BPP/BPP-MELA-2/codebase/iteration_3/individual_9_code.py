import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    i,j = np.indices((n,n))
    remaining = node_constraint - node_attr[:,None] - node_attr[None,:]
    heu = np.where((i != j) & (remaining >= 0), 1/(remaining + 1e-6), 0)
    return heu
    #EVOLVE-END