import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    remaining = node_constraint - node_attr[:, None] - node_attr
    np.fill_diagonal(remaining, -1)  # Prevent self-transitions
    heu = np.where(remaining >= 0, 1/(remaining + 1e-6), 0)
    return heu
    #EVOLVE-END