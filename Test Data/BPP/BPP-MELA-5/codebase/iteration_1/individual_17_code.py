import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint=None):
    #EVOLVE-START
    if node_attr.ndim == 1:
        weights = node_attr
    else:
        weights = node_attr[:, 0]
    
    heur = np.abs(np.subtract.outer(weights, weights))
    heur = 1 / (heur + 1e-6)  # Avoid division by zero
    np.fill_diagonal(heur, 0)  # Prevent self-selection
    
    # Normalize and handle potential all-zero rows
    row_sums = heur.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero for rows with all zeros
    return heur / row_sums
    #EVOLVE-END