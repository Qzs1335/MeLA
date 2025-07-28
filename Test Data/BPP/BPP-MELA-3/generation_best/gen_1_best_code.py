import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            combined = node_attr[i] + node_attr[j]
            weights[i,j] = 1/(1 + abs(node_constraint - combined))
    return weights
    #EVOLVE-END