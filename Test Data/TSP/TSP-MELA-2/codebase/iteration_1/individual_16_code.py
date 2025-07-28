import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    min_val = np.min(distance_matrix)
    norm_matrix = distance_matrix / (min_val + 1e-10)  # Avoid division by zero
    temp = 0.1  # Temperature parameter
    return np.exp(-norm_matrix * temp)
    #EVOLVE-END       
    return Positions