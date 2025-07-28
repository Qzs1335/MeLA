import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    safe_dist = np.maximum(distance_matrix, 1e-6)
    return np.exp(-0.5 * safe_dist) / safe_dist
    #EVOLVE-END       
    return 1 / distance_matrix