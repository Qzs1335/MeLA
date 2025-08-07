import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Ensure node_attr is 2D (n x 1 array)
    sizes = np.array(node_attr).reshape(-1, 1)
    size_diff = np.abs(sizes - sizes.T)
    heuristic = 1 / (1 + size_diff)
    return heuristic * (node_constraint > 0)
    #EVOLVE-END