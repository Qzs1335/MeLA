import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = np.array(node_attr).flatten()  # Ensure sizes is 1D array
    n = sizes.size
    heur = np.outer(sizes, 1/sizes)  # Large items prefer small ones
    np.fill_diagonal(heur, 0)  # Avoid self-pairing
    return heur
    #EVOLVE-END