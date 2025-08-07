import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.array(node_attr)
    if node_attr.ndim == 1:
        sizes = node_attr.reshape(-1,1)
    else:
        sizes = node_attr[:,0].reshape(-1,1)
    n = sizes.shape[0]
    heuristic = 1/(1 + np.abs(sizes - sizes.T))
    heuristic += 0.01 * np.random.rand(n,n)
    np.fill_diagonal(heuristic, 0)
    return heuristic
    #EVOLVE-END