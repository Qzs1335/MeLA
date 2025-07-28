import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10  # Larger epsilon for better stability
    inv_sq = 1 / (distance_matrix**2 + epsilon)
    log_weight = np.log(1 + distance_matrix + epsilon)  # Smoother log
    return inv_sq / log_weight  # Balanced combination
    #EVOLVE-END       
    return 1 / distance_matrix