import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-5
    scale = np.mean(distance_matrix)  
    decay = np.exp(-distance_matrix/scale)
    inv_dist = 1/(distance_matrix + epsilon)**2.0
    return decay * inv_dist
    #EVOLVE-END