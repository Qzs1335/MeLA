import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    min_dist = np.min(distance_matrix[distance_matrix > 0])
    normalized = distance_matrix / min_dist
    return np.log(1 + (1 / (normalized + epsilon)))
    #EVOLVE-END       
    return 1 / distance_matrix