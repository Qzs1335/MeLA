import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    eps = np.finfo(float).eps
    d_norm = distance_matrix / (np.median(distance_matrix) + eps)
    log_weights = 1 / (1 + np.log1p(d_norm))
    hybrid = log_weights*np.exp(-d_norm) + (1-log_weights)/(d_norm + eps)
    return hybrid        
    #EVOLVE-END       
    return 1 / distance_matrix