import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    d_clip = np.clip(distance_matrix, 1e-10, None)  # tighter clipping
    mean_dist = np.median(d_clip)  # robust central tendency
    weights = np.exp(-0.5*d_clip/mean_dist)  # optimized decay
    return weights/d_clip + (1-weights)*np.exp(-d_clip)    
    #EVOLVE-END       
    return 1 / distance_matrix