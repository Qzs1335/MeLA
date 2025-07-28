import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    inv_dist = 1/(distance_matrix + epsilon)
    neighbor_scale = np.min(distance_matrix, axis=1, keepdims=True) + epsilon
    return 0.5*inv_dist + 0.5*np.exp(-distance_matrix/neighbor_scale)
    #EVOLVE-END       
    return 1 / distance_matrix