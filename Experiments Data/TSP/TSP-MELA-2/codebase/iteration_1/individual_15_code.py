import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    min_d = np.min(distance_matrix[np.nonzero(distance_matrix)])
    max_d = np.max(distance_matrix)
    local_avg = np.mean(distance_matrix, axis=1, keepdims=True)
    global_avg = np.mean(distance_matrix)
    ratio = (distance_matrix/local_avg) * (global_avg/distance_matrix)
    return np.log(1 + min_d/distance_matrix) * ratio
    #EVOLVE-END       
    return 1 / distance_matrix