import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = node_attr.flatten()  # Ensure we have a 1D array of sizes
    heuristics = np.outer(sizes, sizes) / node_constraint
    np.fill_diagonal(heuristics, 0)
    return heuristics
    #EVOLVE-END