import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.asarray(node_attr)
    if node_attr.ndim == 1:
        sizes = node_attr.reshape(-1,1)
    else:
        sizes = node_attr[:,0].reshape(-1,1)
    
    n = sizes.shape[0]
    heuristic = 1/(1 + np.abs(sizes - sizes.T))  # Size compatibility
    heuristic += 0.01*np.random.rand(n,n)        # Exploration
    np.fill_diagonal(heuristic, 0)               # No self-pairing
    
    # Handle node_constraint (ensure no division by zero)
    constraint = np.asarray(node_constraint)
    if constraint.ndim == 0:
        constraint = constraint.reshape(1,1)
    return heuristic/(constraint + 1e-10)        # Capacity scaling with small epsilon
    #EVOLVE-END