import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.array(node_attr).reshape(-1,1) if len(node_attr.shape) == 1 else node_attr
    n = node_attr.shape[0]
    sizes = node_attr[:,0].reshape(-1,1)
    normalized = sizes/node_constraint
    heuristic = 1 - np.abs(normalized - normalized.T)
    return heuristic + 0.01*np.random.rand(n,n)
    #EVOLVE-END