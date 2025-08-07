import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    heur = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                ratio_i = node_attr[i,0]/node_attr[i,1]  # weight/volume
                ratio_j = node_attr[j,0]/node_attr[j,1]
                heur[i,j] = ratio_i/(ratio_j + 1e-8) * (node_constraint - node_attr[j,0])
    return heur
    #EVOLVE-END