import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    sizes = node_attr[:,0].reshape(-1,1)
    capacities = node_constraint
    size_ratios = np.minimum(sizes, capacities) / np.maximum(sizes, capacities)
    return np.exp(-5*(1-size_ratios))
    #EVOLVE-END