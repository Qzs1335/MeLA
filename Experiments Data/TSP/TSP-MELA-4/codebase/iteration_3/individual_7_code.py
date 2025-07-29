import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-5
    adaptive_scale = np.mean(distance_matrix)
    decay = np.exp(-distance_matrix/(adaptive_scale + epsilon))
    inv_dist = (1/(np.clip(distance_matrix, epsilon, None)))**1.5
    return 0.7*decay + 0.3*inv_dist
    #EVOLVE-END       
    return 1 / distance_matrix