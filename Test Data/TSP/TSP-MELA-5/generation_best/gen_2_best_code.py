import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    clipped_dist = np.clip(distance_matrix, 1e-5, None)
    return (1/clipped_dist) * np.log(1 + 1/(np.sqrt(clipped_dist) + epsilon))
    #EVOLVE-END       
    return 1 / distance_matrix