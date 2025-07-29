import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    d_clip = np.clip(distance_matrix, 1e-5, None)  # Gradient stability
    scaled_dist = d_clip/np.mean(d_clip)  # Adaptive scaling
    return np.exp(-scaled_dist) + 1e-3*(1/d_clip)  # Gradient-enhanced exploration
    #EVOLVE-END       
    return 1 / distance_matrix