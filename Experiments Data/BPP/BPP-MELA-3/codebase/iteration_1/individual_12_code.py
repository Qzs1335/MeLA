import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Ensure node_attr is 2D array
    node_attr = np.array(node_attr)
    if len(node_attr.shape) == 1:
        node_attr = node_attr.reshape(-1, 1)
    
    sizes = node_attr[:,0] if node_attr.shape[1] > 0 else node_attr.ravel()
    size_diff = np.abs(sizes[:, None] - sizes)
    heuristic = 1 / (1 + size_diff)
    return heuristic / heuristic.max()
    #EVOLVE-END