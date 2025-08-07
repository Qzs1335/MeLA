import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    adaptive_scale = np.mean(distance_matrix)
    d_clip = np.clip(distance_matrix, 1e-5, None)  
    exp_decay = np.exp(-d_clip/(2*adaptive_scale))  
    hybrid = 0.7*exp_decay + 0.3/d_clip  
    return hybrid        
    #EVOLVE-END       
    return 1 / distance_matrix