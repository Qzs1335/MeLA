import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    heu = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                remaining = node_constraint - node_attr[i] - node_attr[j]
                heu[i,j] = 1/(remaining + 1e-6) if remaining >= 0 else 0
    return heu
    #EVOLVE-END