import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    scaled_dist = distance_matrix/np.median(distance_matrix)
    prob_factor = 0.3 + 0.7*np.random.random()
    return prob_factor / (scaled_dist + epsilon)
    #EVOLVE-END       
    return 1 / distance_matrix