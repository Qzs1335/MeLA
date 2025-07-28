import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = np.array(node_attr).flatten()  # Ensure 1D array
    n = len(sizes)
    dist_matrix = np.abs(sizes[:,None] - sizes)
    heuristic = 1/(1 + dist_matrix)
    np.fill_diagonal(heuristic, 0)
    return heuristic/node_constraint
    #EVOLVE-END