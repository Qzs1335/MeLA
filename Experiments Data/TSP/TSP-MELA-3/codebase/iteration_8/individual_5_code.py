import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    epsilon = 1e-10
    d_clip = np.clip(distance_matrix, epsilon, None)
    weights = np.exp(-0.5*d_clip/np.mean(d_clip))  # softer decay
    hybrid = weights/(d_clip**0.8) + (1-weights)/d_clip  # balanced hybrid
    return hybrid        
    #EVOLVE-END       
    return 1 / distance_matrix