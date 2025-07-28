import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Handle both 1D and 2D inputs
    if node_attr.ndim == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:, 0]
    
    # Vectorized calculation
    size_matrix = sizes[:, None] + sizes[None, :]
    residual = node_constraint - size_matrix
    np.fill_diagonal(residual, 0)  # Avoid self-combination
    heuristic = 1 / (np.abs(residual) + 1e-6)
    return heuristic / heuristic.max()
    #EVOLVE-END