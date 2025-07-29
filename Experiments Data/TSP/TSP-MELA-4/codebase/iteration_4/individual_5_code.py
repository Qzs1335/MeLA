import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    scale = np.mean(distance_matrix) + epsilon
    decay = np.exp(-distance_matrix/scale)
    beta = 1.5 + (distance_matrix/scale)
    inv_dist = (1/(distance_matrix + epsilon))**beta
    return decay * inv_dist
    #EVOLVE-END       
    return 1 / distance_matrix