import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    d_clip = np.clip(distance_matrix, 1e-5, None)
    scale = np.exp(-d_clip/np.mean(d_clip))  # adaptive scaling
    return scale * (1.5/d_clip) + (1-scale)/d_clip  # balanced hybrid
    #EVOLVE-END       
    return 1 / distance_matrix