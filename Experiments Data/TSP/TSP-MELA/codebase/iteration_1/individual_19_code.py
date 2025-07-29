import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-10
    min_dist = np.min(distance_matrix[distance_matrix > 0])
    return np.exp(-distance_matrix/(2*min_dist)) + eps
    #EVOLVE-END       
    return 1 / distance_matrix