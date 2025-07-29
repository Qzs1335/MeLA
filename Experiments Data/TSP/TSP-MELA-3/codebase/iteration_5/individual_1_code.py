import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    dynamic_eps = np.maximum(1e-10, 0.01*distance_matrix.mean())
    exp_comp = np.exp(-distance_matrix/2)  # Softer decay
    inv_comp = 1 / (distance_matrix + dynamic_eps)
    hybrid = (0.5 * exp_comp) + (0.5 * inv_comp)
    return hybrid        
    #EVOLVE-END       
    return 1 / distance_matrix