import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    node_sum = node_attr[:, None] + node_attr[None, :]
    remaining = node_constraint - node_sum
    heu = np.divide(1, remaining + 1e-6, where=(remaining >= 0))
    np.fill_diagonal(heu, 0)
    return heu
    #EVOLVE-END