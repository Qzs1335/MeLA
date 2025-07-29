import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    norm_dist = distance_matrix / np.max(distance_matrix)
    return np.log(1 + (1 / (norm_dist + epsilon)))
    #EVOLVE-END
    return 1 / distance_matrix