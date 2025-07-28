import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    heur = np.zeros((n, n))
    for i in range(n):
        remaining_cap = node_constraint - node_attr[i]
        heur[i] = 1/(remaining_cap + 1e-6)  # Avoid division by zero
        heur[i] += np.random.rand(n)*0.1  # Small random component
        heur[i] /= np.max(heur[i])  # Normalize
    return heur
    #EVOLVE-END