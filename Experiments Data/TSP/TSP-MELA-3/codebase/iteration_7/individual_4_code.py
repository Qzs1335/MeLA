import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    d_clip = np.clip(distance_matrix, 1e-5, None)  
    return np.exp(1/d_clip) + 1/d_clip        
    #EVOLVE-END       
    return 1 / distance_matrix