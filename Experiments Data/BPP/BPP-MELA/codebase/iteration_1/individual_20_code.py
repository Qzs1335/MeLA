import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Ensure node_attr is treated as 1D array
    sizes = np.asarray(node_attr).flatten()  # Convert to 1D array and flatten
    
    # Calculate heuristic matrix
    heuristic = np.outer(sizes, sizes) / (node_constraint**2)
    np.fill_diagonal(heuristic, 0)  # Avoid self-selection
    return heuristic
    #EVOLVE-END