import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Convert inputs to numpy arrays if they aren't already
    node_attr = np.asarray(node_attr)
    node_constraint = np.asarray(node_constraint)
    
    # Reshape if node_attr is 1D
    if node_attr.ndim == 1:
        sizes = node_attr.reshape(-1, 1)
    else:
        sizes = node_attr[:, 0].reshape(-1, 1)
    
    size_ratios = np.outer(sizes.flatten(), 1/sizes.flatten())
    capacity_ratios = np.outer(sizes.flatten(), 1/node_constraint.flatten())
    return 0.6*size_ratios + 0.4*capacity_ratios
    #EVOLVE-END