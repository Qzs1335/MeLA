import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    eps = 1e-10
    d_clip = np.clip(distance_matrix, 1e-5, None)
    weights = np.exp(-d_clip/np.mean(d_clip**0.5))  # Better distance sensitivity 
    inv_d = 1/(d_clip**0.5 + eps)                  # Smoother gradient
    hybrid = weights*np.exp(inv_d) + (1-weights)*inv_d
    return hybrid        
    #EVOLVE-END       
    return 1 / distance_matrix