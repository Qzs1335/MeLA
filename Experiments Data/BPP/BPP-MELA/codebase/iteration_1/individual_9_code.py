import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.asarray(node_attr)
    if node_attr.ndim == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:,0]
    normalized_sizes = sizes / np.max(sizes)
    heuristics = np.outer(normalized_sizes, normalized_sizes)
    np.fill_diagonal(heuristics, 0)
    return heuristics
    #EVOLVE-END