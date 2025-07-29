import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-10
    decay_factor = 0.5
    norm_dist = distance_matrix / np.max(distance_matrix)
    return np.exp(-decay_factor * norm_dist) + eps
    #EVOLVE-END
    return 1 / distance_matrix