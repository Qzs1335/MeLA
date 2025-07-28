import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = len(node_attr)
    # Handle 1D or 2D node_attr
    weights = node_attr if node_attr.ndim == 1 else node_attr[:,0]
    # Handle scalar or array node_constraint
    capacity = node_constraint if np.isscalar(node_constraint) else node_constraint[0]
    
    heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                remaining = capacity - weights[i] - weights[j]
                heuristic[i,j] = 1/(1 + abs(remaining)) + 0.1*np.random.rand()
    np.fill_diagonal(heuristic, 0)
    return heuristic/heuristic.max()
    #EVOLVE-END