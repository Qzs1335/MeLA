import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Handle both 1D and 2D node_attr cases
    sizes = node_attr if len(node_attr.shape) == 1 else node_attr[:, 0]
    n = sizes.shape[0]
    heuristic = np.abs(np.subtract.outer(sizes, sizes))
    heuristic = 1/(1 + heuristic)  # Inverse relationship
    heuristic += 0.01 * np.random.rand(n, n)  # Exploration
    np.fill_diagonal(heuristic, 0)  # No self-transition
    return heuristic
    #EVOLVE-END