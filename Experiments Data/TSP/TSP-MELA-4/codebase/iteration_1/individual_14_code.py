import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    normalized_dist = distance_matrix / np.max(distance_matrix)
    return np.log1p(1 / (normalized_dist + epsilon))
    #EVOLVE-END
    return 1 / distance_matrix