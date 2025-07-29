import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    dists = np.linalg.norm(node_attr[:,None] - node_attr, axis=2)
    heur = 1/(1 + dists) * (1 + node_constraint[:,None])
    heur += 0.01 * np.random.rand(n,n)
    return heur
    #EVOLVE-END