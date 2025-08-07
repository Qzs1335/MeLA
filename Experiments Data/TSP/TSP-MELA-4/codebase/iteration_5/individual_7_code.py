import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    scale = np.median(distance_matrix) 
    beta = 1.2 + 0.5*np.log1p(scale)  # optimized dynamic exponent
    decay = np.exp(-distance_matrix/scale)
    inv_dist = (1/(distance_matrix + epsilon))**beta
    return decay * inv_dist * (1 + np.log1p(1/(distance_matrix + epsilon)))
    #EVOLVE-END       
    return 1 / distance_matrix