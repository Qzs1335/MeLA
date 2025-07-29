import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.asarray(node_attr)
    if node_attr.ndim == 1:
        node_attr = node_attr.reshape(-1, 1)
    
    n = node_attr.shape[0]
    sizes = node_attr[:, 0] if node_attr.shape[1] > 0 else np.zeros(n)
    heuristic = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                combined = sizes[i] + sizes[j]
                heuristic[i,j] = 1/(abs(node_constraint - combined) + 1e-9)
    
    np.fill_diagonal(heuristic, 0)
    return heuristic
    #EVOLVE-END