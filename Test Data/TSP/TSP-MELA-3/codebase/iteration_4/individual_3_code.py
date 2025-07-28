import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-5
    bounded_dist = np.clip(distance_matrix, epsilon, 1/epsilon)
    return 1 / bounded_dist
    #EVOLVE-END       
    return 1 / distance_matrix