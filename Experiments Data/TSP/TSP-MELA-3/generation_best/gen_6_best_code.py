import numpy as np
import numpy as np  
def heuristics_v2(distance_matrix):  
    #EVOLVE-START    
    epsilon = 1e-10  
    d_clip = np.clip(distance_matrix, epsilon, None)  
    mean_dist = np.mean(d_clip)  
    weights = np.exp(-d_clip/mean_dist)  
    hybrid = (weights*np.exp(1/d_clip) + (1-weights)/(d_clip + epsilon))/np.max(d_clip)  
    return hybrid  
    #EVOLVE-END       
    return 1 / distance_matrix