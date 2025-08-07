import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = np.mean(distance_matrix)/100  # dynamic epsilon
    decay_scale = 5  # optimized from 10
    normalized = np.exp(-distance_matrix/decay_scale) * ((1/(distance_matrix + epsilon))**2.0)
    return normalized/np.max(normalized)  # scale to [0,1]
    #EVOLVE-END       
    return 1 / distance_matrix