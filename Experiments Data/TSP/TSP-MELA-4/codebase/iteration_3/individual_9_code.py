import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    adaptive_scale = np.mean(distance_matrix) + 1e-7
    decay = np.exp(-distance_matrix/adaptive_scale)
    inv_dist = (1/(distance_matrix + 1e-7))**1.5
    return decay * inv_dist
    #EVOLVE-END       
    return 1 / distance_matrix