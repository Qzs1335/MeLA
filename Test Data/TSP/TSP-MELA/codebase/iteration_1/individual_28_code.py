import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-6
    scaled_dist = distance_matrix / np.median(distance_matrix)
    return np.exp(-0.5 * scaled_dist) / (scaled_dist + epsilon)
    #EVOLVE-END
    return 1 / distance_matrix