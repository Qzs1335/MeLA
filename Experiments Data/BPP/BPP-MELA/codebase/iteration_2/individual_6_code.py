import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = node_attr if node_attr.ndim == 1 else node_attr[:,0]
    n = len(sizes)
    size_matrix = sizes[:,None] + sizes
    remaining = node_constraint - size_matrix
    valid = (remaining >= 0).astype(float)
    heuristic = valid * (1/(abs(remaining)+1e-6) + sizes[:,None]/node_constraint)
    return heuristic/(heuristic.max()+1e-6)
    #EVOLVE-END