import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    weights = node_attr[:,0]/np.max(node_attr[:,0])
    constraints = node_attr[:,1]/np.max(node_attr[:,1])
    heuristic = np.outer(weights, constraints) * (1 + np.abs(np.subtract.outer(weights, constraints)))
    return heuristic/np.max(heuristic)
    #EVOLVE-END