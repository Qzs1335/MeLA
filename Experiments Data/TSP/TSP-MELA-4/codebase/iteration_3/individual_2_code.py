import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-7  # tighter stability
    mean_dist = np.mean(distance_matrix)
    beta = 2.0 - np.log1p(mean_dist)  # dynamic parameter
    decay = np.exp(-distance_matrix/(mean_dist + epsilon))
    inv_dist = (1/(distance_matrix + epsilon))**beta
    return decay * inv_dist * np.log1p(1/distance_matrix)
    #EVOLVE-END       
    return 1 / distance_matrix