import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    visited = np.zeros_like(distance_matrix[0])
    scaled_dist = np.log1p(distance_matrix)
    freq_factor = 1/(1 + visited)
    return freq_factor / (scaled_dist + 1e-10)
    #EVOLVE-END
    return 1 / distance_matrix