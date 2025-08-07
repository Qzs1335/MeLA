import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10 * np.median(distance_matrix)
    adaptive_scale = np.percentile(distance_matrix, 75)
    dynamic_beta = 1.4 + 0.3*np.cos(adaptive_scale/2)
    power_law = (distance_matrix + epsilon)**-0.7
    sigmoid = 1/(1+np.exp(-distance_matrix/adaptive_scale)) 
    return power_law * sigmoid * (1 + np.log1p(1/(distance_matrix + epsilon))) 
    #EVOLVE-END       
    return 1 / distance_matrix