import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.asarray(node_attr)
    if node_attr.ndim == 1:
        node_attr = node_attr.reshape(-1,1)
    n = node_attr.shape[0]
    sizes = node_attr[:,0].reshape(-1,1)
    return np.tile(sizes, (1,n)) + np.tile(sizes.T, (n,1)) - node_constraint
    #EVOLVE-END