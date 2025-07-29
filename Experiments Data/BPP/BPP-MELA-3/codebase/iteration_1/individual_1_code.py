import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.array(node_attr)
    if len(node_attr.shape) == 1:
        node_attr = node_attr.reshape(-1, 1)
    n = node_attr.shape[0]
    sizes = node_attr[:,0].reshape(-1,1)
    heuristic = 1/(1e-6 + np.abs(sizes - sizes.T))
    heuristic += 0.01*np.random.rand(n,n)
    np.fill_diagonal(heuristic, 0)
    return heuristic
    #EVOLVE-END