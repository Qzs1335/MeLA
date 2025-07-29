import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    heuristic = 1 / (node_constraint[:, None] + 1e-6)  # Inverse constraints with epsilon for stability
    heuristic += np.random.uniform(0, 0.1, (n, n))  # Small random exploration factor
    np.fill_diagonal(heuristic, 0)  # Prevent self-selection
    return heuristic
    #EVOLVE-END