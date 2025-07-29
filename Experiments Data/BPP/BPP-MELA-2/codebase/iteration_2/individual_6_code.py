import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    attr_diff = np.abs(node_attr[:,None] - node_attr[None,:])
    remaining = node_constraint - (node_attr[:,None] + node_attr[None,:])
    heu = (1/(remaining + 1e-6)) * (1/(attr_diff + 1e-6))
    np.fill_diagonal(heu, 0)
    heu[remaining < 0] = 0
    return heu
    #EVOLVE-END