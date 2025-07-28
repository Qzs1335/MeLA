import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Handle both 1D and 2D input arrays
    sizes = node_attr if node_attr.ndim == 1 else node_attr[:,0]
    size_matrix = np.abs(sizes[:,None] - sizes[None,:])
    norm_size = 1 - (size_matrix / np.max(size_matrix))
    capacity_ratio = np.minimum(sizes[:,None], sizes[None,:]) / node_constraint
    return 0.7*norm_size + 0.3*capacity_ratio
    #EVOLVE-END