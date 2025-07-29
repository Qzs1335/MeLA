import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    if len(node_attr.shape) == 1:
        sizes = node_attr
    else:
        sizes = node_attr[:, 0]
    heuristic = np.outer(sizes, 1/sizes)
    heuristic = heuristic / heuristic.max()
    return heuristic
    #EVOLVE-END