import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10  # smaller epsilon for better precision
    adaptive_temp = np.median(distance_matrix)  # dynamic scaling
    return np.exp(-distance_matrix/adaptive_temp) * (1/(distance_matrix + epsilon))
    #EVOLVE-END       
    return 1 / distance_matrix