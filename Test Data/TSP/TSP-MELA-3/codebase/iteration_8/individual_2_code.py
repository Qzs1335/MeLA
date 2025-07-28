import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    epsilon = 1e-10
    d_scaled = distance_matrix/(np.std(distance_matrix)+epsilon)  # unit variance scaling
    d_clip = np.clip(d_scaled, 1e-5, None)
    weights = 0.5*np.exp(-d_clip)  # reduced weight sensitivity
    hybrid = weights*np.exp(1/d_clip) + (1-weights)/d_clip
    return hybrid        
    #EVOLVE-END       
    return 1 / distance_matrix