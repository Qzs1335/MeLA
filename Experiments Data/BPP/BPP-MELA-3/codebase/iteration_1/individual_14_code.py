import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Handle both 1D and 2D input cases
    sizes = node_attr if len(node_attr.shape) == 1 else node_attr[:, 0]
    normalized_sizes = sizes / node_constraint
    h_matrix = np.outer(normalized_sizes, normalized_sizes)
    return np.exp(-h_matrix)  # Inverse relationship for smaller items
    #EVOLVE-END