import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    d_safe = np.maximum(distance_matrix, 1e-10)
    w = np.exp(-distance_matrix/distance_matrix.mean()) 
    return w/d_safe + (1-w)*np.exp(1/d_safe)
    #EVOLVE-END       
    return 1 / distance_matrix