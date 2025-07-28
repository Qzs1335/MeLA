import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    eps = 1e-10
    d_clip = np.clip(distance_matrix, 1e-3, None)
    norm_dist = d_clip / np.max(d_clip)  
    weights = 0.5 * np.exp(-2 * norm_dist) 
    hybrid = weights * (np.log(1 + 1/d_clip)/2 + np.exp(1/d_clip)/2) + (1 - weights)/d_clip
    return np.nan_to_num(hybrid, nan=eps)        
    #EVOLVE-END       
    return 1 / distance_matrix