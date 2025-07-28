import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    epsilon = 1e-10
    d_clip = np.clip(distance_matrix, 1e-5, None)
    d_normalized = d_clip/np.mean(d_clip)
    weights = np.exp(-d_normalized)
    hybrid = 0.6*(weights*(np.log10(5+1/d_clip)) + (1-weights)/d_clip) - (0.01*d_normalized)
    return hybrid
    #EVOLVE-END