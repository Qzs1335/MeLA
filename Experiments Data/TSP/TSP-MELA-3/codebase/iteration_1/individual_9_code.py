import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-10
    log_factor = 0.5 * np.log(1 + distance_matrix)
    rand_weights = 1 + 0.1 * np.random.rand(*distance_matrix.shape)
    return rand_weights / (distance_matrix + eps) * log_factor
    #EVOLVE-END       
    return 1 / distance_matrix