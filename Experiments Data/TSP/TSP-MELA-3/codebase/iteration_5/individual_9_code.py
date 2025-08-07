import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    epsilon = 1e-10
    log_scale = np.log(1 + distance_matrix)
    norm_dist = distance_matrix / (np.max(distance_matrix) + epsilon)
    hybrid = (0.6 * np.exp(1/(norm_dist + epsilon))) + (0.4 / (log_scale + epsilon))
    return hybrid        
    #EVOLVE-END       
    return 1 / distance_matrix