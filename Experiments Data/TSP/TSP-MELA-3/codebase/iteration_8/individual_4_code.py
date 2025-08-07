import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    epsilon = 1e-10
    d_clip = np.clip(distance_matrix, 1e-5, None)
    d_stats = np.mean(d_clip) + np.std(d_clip)  # mean+std as scaling factor
    weights = np.exp(-d_clip/d_stats)  # more adaptive weights
    hybrid = weights*(1 + np.exp(1/d_clip)) + (1-weights)/d_clip  # enhanced mix
    return hybrid / np.max(hybrid)  # normalized output
    #EVOLVE-END       
    return 1 / distance_matrix