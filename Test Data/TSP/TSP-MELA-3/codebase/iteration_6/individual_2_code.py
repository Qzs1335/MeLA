import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    d_clip = np.clip(distance_matrix, 1e-7, None)  # refined clipping
    p_normalize = np.percentile(d_clip, 75)  # use 75th percentile
    weights = 0.5*(1 + np.tanh((p_normalize-d_clip)/p_normalize))
    hybrid = weights*np.exp(1/d_clip) + (1-weights)/d_clip
    return np.nan_to_num(hybrid)      
    #EVOLVE-END       
    return 1 / distance_matrix