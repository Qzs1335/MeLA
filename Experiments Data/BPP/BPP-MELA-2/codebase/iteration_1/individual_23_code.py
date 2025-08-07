import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    # Convert inputs to numpy arrays if they aren't already
    node_attr = np.asarray(node_attr)
    node_constraint = np.asarray(node_constraint)
    
    n = node_attr.shape[0]
    dist = np.zeros((n, n))
    
    # Calculate distance matrix
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i,j] = 1/(1 + np.linalg.norm(node_attr[i]-node_attr[j]))
            else:
                dist[i,j] = 1
                
    # Handle scalar constraint case
    if node_constraint.ndim == 0:
        return dist * (2 * node_constraint)/node_constraint
    else:
        return dist * (node_constraint[:,None] + node_constraint)/node_constraint.max()
    #EVOLVE-END