import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    heu = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                weight_diff = abs(node_attr[i] - node_attr[j])
                heu[i,j] = 1/(1 + weight_diff) * (node_constraint - weight_diff)
    return heu
    #EVOLVE-END