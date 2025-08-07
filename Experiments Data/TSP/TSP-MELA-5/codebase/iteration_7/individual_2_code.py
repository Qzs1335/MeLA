import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    scale_factor = np.median(distance_matrix)
    epsilon = max(1e-15, scale_factor * 1e-6)
    smoothed_dist = distance_matrix + epsilon
    return (1 / smoothed_dist) * np.log1p(1 / smoothed_dist)
    #EVOLVE-END       
    return 1 / distance_matrix