import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    sizes = node_attr if node_attr.ndim == 1 else node_attr[:,0]
    n = len(sizes)
    size_pairs = sizes[:,None] + sizes
    inv_diff = 1/(abs(size_pairs - node_constraint) + 1e-6)
    ratio = np.minimum(sizes[:,None]/node_constraint, sizes/node_constraint)
    return (0.7*inv_diff + 0.3*ratio)/np.max(0.7*inv_diff + 0.3*ratio)
    #EVOLVE-END