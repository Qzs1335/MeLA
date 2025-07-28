import numpy as np
import numpy as np  
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    d_clip = np.clip(distance_matrix, 1e-3, None)  
    weights = 1 / (1 + np.exp(d_clip - np.median(d_clip)))  # sigmoid-based weights
    hybrid = weights * (1/(d_clip + 1e-8)) + (1 - weights) * np.exp(-d_clip)  
    return hybrid        
    #EVOLVE-END       
    return 1 / distance_matrix