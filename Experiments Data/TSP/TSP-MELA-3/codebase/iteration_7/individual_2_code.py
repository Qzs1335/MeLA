import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    dist_mean = np.mean(distance_matrix)
    d_clip = np.clip(distance_matrix, max(1e-5, 0.1*dist_mean), None)
    log_term = np.log(1 + d_clip/dist_mean)
    exp_term = np.exp(-d_clip/dist_mean)
    hybrid = (exp_term + 1/(1 + log_term)) / d_clip
    return hybrid        
    #EVOLVE-END       
    return 1 / distance_matrix