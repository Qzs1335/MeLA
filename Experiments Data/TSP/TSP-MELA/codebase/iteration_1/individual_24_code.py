import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    nn_influence = np.sum(1/distance_matrix, axis=1, keepdims=True)
    log_scale = np.log(distance_matrix + 1e-10)
    return (1 / (distance_matrix + 1e-7)) * (nn_influence / log_scale)
    #EVOLVE-END       
    return 1 / distance_matrix