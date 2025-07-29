import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    scaled_dist = np.log(distance_matrix + epsilon)
    return (1 / (scaled_dist + 1)) * (1 / np.max(scaled_dist))
    #EVOLVE-END       
    return 1 / distance_matrix