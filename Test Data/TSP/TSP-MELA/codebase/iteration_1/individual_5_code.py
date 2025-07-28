import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    normalized_dist = distance_matrix / np.max(distance_matrix)
    return np.exp(-2 * normalized_dist) / distance_matrix
    #EVOLVE-END
    return 1 / distance_matrix