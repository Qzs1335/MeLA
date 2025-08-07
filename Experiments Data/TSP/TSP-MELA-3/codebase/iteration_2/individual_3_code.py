import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    nearest_scaler = np.min(distance_matrix, axis=1, keepdims=True)
    return nearest_scaler / (distance_matrix + epsilon)
    #EVOLVE-END       
    return 1 / distance_matrix