import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Ensure node_attr is 2D with shape (n_nodes, n_features)
    node_attr = np.atleast_2d(node_attr)
    if node_attr.shape[0] == 1:  # If input was 1D, transpose to column vector
        node_attr = node_attr.T
    n = node_attr.shape[0]
    sizes = node_attr[:, 0] if node_attr.shape[1] > 0 else np.ones(n)
    # Create size-based heuristic matrix
    heuristic = np.outer(sizes, 1/sizes)
    # Add exploration noise
    noise = 0.1 * np.random.rand(n, n)
    heuristic = (heuristic + noise) / (heuristic + noise).sum(axis=1, keepdims=True)
    return heuristic
    #EVOLVE-END