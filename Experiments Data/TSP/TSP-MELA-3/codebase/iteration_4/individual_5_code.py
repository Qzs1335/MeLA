import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    epsilon = 1e-10
    hybrid = (0.7 * np.exp(1/(distance_matrix + epsilon))) + (0.3 / (distance_matrix + epsilon))
    return hybrid        
    #EVOLVE-END       
    return 1 / distance_matrix