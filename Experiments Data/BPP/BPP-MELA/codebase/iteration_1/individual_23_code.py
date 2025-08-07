import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    if len(node_attr.shape) == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:,0]
    n = sizes.shape[0]
    heur = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                size_diff = abs(sizes[i]-sizes[j])
                heur[i,j] = 1/(1+size_diff) if sizes[i]+sizes[j] <= node_constraint else 0
    np.fill_diagonal(heur, 0)
    return heur
    #EVOLVE-END