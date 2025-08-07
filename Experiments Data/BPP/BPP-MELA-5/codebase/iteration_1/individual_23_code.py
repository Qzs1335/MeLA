import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = np.array(node_attr).flatten()  # Ensure 1D array
    size_diff = np.abs(sizes[:, None] - sizes[None, :])
    # Add small epsilon to avoid division by zero
    heuristic = 1 / (1 + size_diff/(node_constraint + 1e-10))
    return heuristic
    #EVOLVE-END