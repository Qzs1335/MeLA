import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    d_clip = np.clip(distance_matrix, 1e-5, None)
    avg_dist = np.mean(d_clip)
    w = np.exp(-(d_clip/avg_dist)**0.5)  # Smoother weight decay
    return w*np.exp(1/d_clip) + 0.5*(1-w)/d_clip  # Balanced hybrid
    #EVOLVE-END       
    return 1 / distance_matrix