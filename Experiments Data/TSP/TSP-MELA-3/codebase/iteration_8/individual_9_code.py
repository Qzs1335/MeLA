import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    epsilon = 1e-10
    d_clip = np.clip(distance_matrix, 1e-5, None)
    mean_dist = np.mean(d_clip)
    weights = 1/(1+np.exp(3*(d_clip/mean_dist-1)))  # Sigmoid transition
    hybrid = weights*3.5*np.exp(1/d_clip) + (1-weights)*0.7/d_clip  
    hybrid *= 1 + 1e-3*np.random.rand(*d_clip.shape)  # Small noise
    return np.nan_to_num(hybrid, nan=0)        
    #EVOLVE-END       
    return 1 / distance_matrix