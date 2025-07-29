import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    heuristic = 1 / (node_constraint[:, None] + node_constraint[None, :] + 1e-10)
    np.fill_diagonal(heuristic, 0)
    return heuristic / heuristic.max()
    #EVOLVE-END