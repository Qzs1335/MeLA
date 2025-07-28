import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8  
    log_scale = np.log1p(np.median(distance_matrix))
    dynamic_beta = 1.2 + 0.5 * np.log1p(log_scale)  
    decay = np.exp(-distance_matrix/(log_scale + epsilon))  
    inv_dist = (1/(distance_matrix + epsilon))**dynamic_beta  
    return decay * inv_dist * np.log1p(1/(distance_matrix + epsilon))  
    #EVOLVE-END       
    return 1 / distance_matrix