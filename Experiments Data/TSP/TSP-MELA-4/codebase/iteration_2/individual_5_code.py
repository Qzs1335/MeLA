import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-5
    temp = np.mean(distance_matrix)  # adaptive scaling
    decay = np.exp(-distance_matrix/(temp + epsilon))
    inv_dist = (1/(distance_matrix + epsilon))**2.5  # tuned beta
    return decay * inv_dist
    #EVOLVE-END       
    return 1 / distance_matrix