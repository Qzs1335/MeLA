import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = node_attr if node_attr.ndim == 1 else node_attr[:,0]
    n = len(sizes)
    size_matrix = sizes.reshape(-1,1) + sizes
    diff = np.abs(size_matrix - node_constraint)
    ratio = size_matrix / (node_constraint + 1e-6)
    heuristic = (1/(diff + 1e-6)) * ratio
    return heuristic / heuristic.max()
    #EVOLVE-END