import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = node_attr[:, 0] if node_attr.ndim > 1 else node_attr
    n = len(sizes)
    size_matrix = np.abs(np.add.outer(sizes, sizes) - node_constraint)
    capacity_ratio = np.minimum(np.add.outer(sizes, sizes)/node_constraint, 1)
    heuristic = capacity_ratio / (size_matrix + 1e-6)
    return heuristic / heuristic.max()
    #EVOLVE-END