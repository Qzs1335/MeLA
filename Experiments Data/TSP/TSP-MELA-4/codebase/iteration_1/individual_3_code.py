import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    log_scale = np.log1p(distance_matrix)
    random_factor = 1 + 0.1 * np.random.rand(*distance_matrix.shape)
    return random_factor / (distance_matrix + epsilon * (distance_matrix == 0)) * np.exp(-log_scale)
    #EVOLVE-END       
    return 1 / distance_matrix