import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    n = node_attr.shape[0]
    base_heuristic = 1 / (node_constraint[:, None] + node_constraint[None, :] + 1e-10)
    noise = 0.1 * np.random.rand(n, n)
    return base_heuristic * (1 + noise)
    #EVOLVE-END