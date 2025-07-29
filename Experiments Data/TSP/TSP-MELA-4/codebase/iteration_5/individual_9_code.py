import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    scale = np.median(distance_matrix) + 0.5*np.std(distance_matrix)
    beta = 1.2 + np.log1p(scale) 
    decay = np.exp(-distance_matrix/(scale + epsilon))
    inv_dist = (1/(distance_matrix + epsilon))**beta
    return decay * inv_dist * np.log1p(scale/(distance_matrix + epsilon))
    #EVOLVE-END       
    return 1 / distance_matrix