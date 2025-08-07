import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.asarray(node_attr)
    sizes = node_attr.reshape(-1) if node_attr.ndim == 1 else node_attr[:,0]
    n = sizes.size
    size_ratios = np.outer(sizes, 1/sizes)
    constraint_denominator = (sizes[:,None] + sizes)
    constraint_denominator = np.maximum(constraint_denominator, 1e-6)  # avoid division by zero
    capacity_ratios = np.minimum(size_ratios, node_constraint/constraint_denominator)
    heur = capacity_ratios * (0.9 + 0.1*np.random.rand(n,n))
    np.fill_diagonal(heur, 0)
    return heur
    #EVOLVE-END