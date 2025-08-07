import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    epsilon = 1e-20
    d_clip = np.clip(distance_matrix, epsilon, None)
    scale = np.median(d_clip)
    return np.exp(-d_clip/scale) + np.exp(1/d_clip)/scale        
    #EVOLVE-END       
    return 1 / distance_matrix