import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    heur = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                fit = node_attr[i] + node_attr[j]
                heur[i,j] = 1/(1 + abs(node_constraint - fit))
    np.fill_diagonal(heur, 0)
    return heur + 0.1*np.random.rand(n,n)
    #EVOLVE-END