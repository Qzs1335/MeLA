import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = node_attr[:,0].reshape(-1,1)
    size_diff = np.abs(sizes - sizes.T)
    constraint_diff = np.abs(node_constraint - node_constraint.T)
    return np.exp(-0.5*(size_diff + constraint_diff))
    #EVOLVE-END