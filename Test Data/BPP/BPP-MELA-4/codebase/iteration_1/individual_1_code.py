import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = np.array(node_attr).flatten()  # Ensure 1D array
    n = sizes.size
    heuristic = np.outer(sizes, sizes)     # Larger items get higher priority
    np.fill_diagonal(heuristic, 0)        # Avoid self-pairing
    return heuristic/sizes.sum()           # Normalize
    #EVOLVE-END