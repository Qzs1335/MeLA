import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    epsilon = 1e-10
    progress_factor = min(1.0, np.mean(distance_matrix)/10)  # [0,1] normalization
    exploit_weight = 0.3 + 0.5*progress_factor 
    hybrid = (exploit_weight * np.exp(1/(distance_matrix + epsilon))) + ((1-exploit_weight)/(distance_matrix + epsilon))
    return hybrid        
    #EVOLVE-END       
    return 1 / distance_matrix