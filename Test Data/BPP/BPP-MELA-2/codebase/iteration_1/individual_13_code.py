import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr) if node_attr.ndim == 1 else node_attr.shape[0]
    sizes = node_attr.reshape(-1,1) if node_attr.ndim == 1 else node_attr[:,0].reshape(-1,1)
    # Size compatibility matrix (larger items prefer smaller companions)
    heuristic = 1/(1 + np.abs(sizes - sizes.T)/node_constraint)
    # Add exploration noise
    heuristic *= (1 + 0.1*np.random.rand(n,n))
    np.fill_diagonal(heuristic, 0)  # No self-selection
    return heuristic
    #EVOLVE-END