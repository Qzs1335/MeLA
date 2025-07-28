import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    node_sum = node_attr[:, None] + node_attr
    remaining = node_constraint - node_sum
    heu = 1/(remaining + 1e-6)
    heu = np.where((remaining < 0) | (np.eye(n, dtype=bool)), 0, heu)
    return heu
    #EVOLVE-END