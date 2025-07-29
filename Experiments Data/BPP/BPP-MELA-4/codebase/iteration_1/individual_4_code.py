import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.array(node_attr)
    if node_attr.ndim == 1:
        node_attr = node_attr.reshape(-1, 1)
    n = node_attr.shape[0]
    sizes = node_attr[:,0].reshape(-1,1) if node_attr.shape[1] > 0 else node_attr.reshape(-1,1)
    size_ratio = sizes / (sizes.T + 1e-8)
    heuristic = 0.5 * size_ratio + 0.5 * np.random.rand(n,n)
    return heuristic / heuristic.max()
    #EVOLVE-END