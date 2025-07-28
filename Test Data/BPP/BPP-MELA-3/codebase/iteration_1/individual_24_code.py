import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.asarray(node_attr)
    if node_attr.ndim == 1:
        node_attr = node_attr.reshape(-1, 1)
    sizes = node_attr[:,0] if node_attr.shape[1] > 0 else node_attr
    sizes = sizes.reshape(-1,1)
    size_diff = np.abs(sizes - sizes.T)
    heuristic = np.exp(-size_diff/node_constraint)
    np.fill_diagonal(heuristic, 0)
    return heuristic
    #EVOLVE-END