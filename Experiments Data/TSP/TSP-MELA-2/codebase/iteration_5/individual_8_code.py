import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    scaled_dist = distance_matrix / (np.median(distance_matrix, axis=1)[:, None] + eps)
    sigmoid = 1 / (1 + np.exp(scaled_dist))
    return sigmoid / (distance_matrix + eps)
    #EVOLVE-END       
    return 1 / distance_matrix