import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    adaptive_scale = np.median(distance_matrix)
    inv_dist = 1/(distance_matrix + epsilon)
    decay = np.exp(-distance_matrix/adaptive_scale)
    return (inv_dist * decay) / np.max(inv_dist * decay)
    #EVOLVE-END       
    return 1 / distance_matrix