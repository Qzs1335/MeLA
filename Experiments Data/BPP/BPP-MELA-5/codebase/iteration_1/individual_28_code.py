import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr) if len(node_attr.shape) == 1 else node_attr.shape[0]
    weights = node_attr if len(node_attr.shape) == 1 else node_attr[:,0]
    heur = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                remaining = node_constraint - weights[j]
                heur[i,j] = np.exp(-abs(weights[i]-remaining)/node_constraint)
    np.fill_diagonal(heur, 0)
    return heur
    #EVOLVE-END