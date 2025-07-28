import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = np.array(node_attr).reshape(-1,1)  # Ensure column vector
    max_size = sizes.max()
    heuristic = np.exp(sizes / max_size) @ np.exp(sizes.T / max_size)
    np.fill_diagonal(heuristic, 0)
    return heuristic
    #EVOLVE-END