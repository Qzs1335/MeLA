import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    mean_distance = np.mean(distance_matrix)
    return np.exp(-distance_matrix / (mean_distance + epsilon)) + 1/(distance_matrix + epsilon)
    #EVOLVE-END       
    return 1 / distance_matrix