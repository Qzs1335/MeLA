import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    epsilon = 1e-10
    sigmoid = 1 / (1 + np.exp(-(distance_matrix - np.median(distance_matrix))))
    dyn_weight = 0.3 + 0.4*sigmoid  # Varies between 0.3-0.7
    hybrid = (dyn_weight * np.exp(1/(distance_matrix + epsilon))) + ((1-dyn_weight) / (distance_matrix + epsilon))
    return hybrid        
    #EVOLVE-END       
    return 1 / distance_matrix