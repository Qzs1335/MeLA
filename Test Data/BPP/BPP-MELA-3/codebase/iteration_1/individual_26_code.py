import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Ensure node_attr is 2D (n,1) array
    sizes = np.array(node_attr).reshape(-1, 1)
    n = sizes.shape[0]
    similarity = 1 - np.abs(sizes - sizes.T)/node_constraint
    np.fill_diagonal(similarity, 0)
    return similarity
    #EVOLVE-END