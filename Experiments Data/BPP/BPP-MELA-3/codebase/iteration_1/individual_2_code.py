import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i,j] = 1/(1 + abs(node_attr[i] - node_attr[j]))
    return dist_matrix/node_constraint
    #EVOLVE-END