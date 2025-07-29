import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    remaining = node_constraint - node_attr[:, None] - node_attr[None, :]
    heu = 1 / (remaining + 1e-6)
    heu = np.where((remaining >= 0) & (~np.eye(n, dtype=bool)), heu, 0)
    return heu
    #EVOLVE-END