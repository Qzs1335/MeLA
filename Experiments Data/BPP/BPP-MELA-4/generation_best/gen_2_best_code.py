import numpy as np
import numpy as np 
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    node_attr = np.asarray(node_attr)
    sizes = node_attr.flatten() if node_attr.ndim == 1 else node_attr[:,0]
    n = len(sizes)
    i,j = np.indices((n,n))
    mask = (i != j)
    h = np.where(mask, 1/(np.abs(sizes[i]+sizes[j]-node_constraint)+1e-6), 0)
    return h
    #EVOLVE-END