import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    small_const = 1e-5
    distance_matrix = distance_matrix + small_const  # Smoothing
    log_dist = np.log(1 + distance_matrix)
    heuristic = np.exp(-log_dist * 0.5)  # Enhanced shaping
    #EVOLVE-END       
    return heuristic