import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    epsilon = 1e-10
    d_clip = np.clip(distance_matrix, 1e-5, None)  
    mean_d = np.mean(d_clip)
    w_exp = np.exp(-d_clip/mean_d)  
    hybrid = w_exp * (np.exp(1/d_clip) + (1/d_clip))  # unified exponential-inverse
    return hybrid + epsilon       
    #EVOLVE-END       
    return 1 / distance_matrix