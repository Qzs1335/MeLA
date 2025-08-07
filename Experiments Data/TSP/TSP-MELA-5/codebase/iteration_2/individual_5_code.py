import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    clipped_dist = np.clip(distance_matrix, epsilon, None)
    return np.log(1 + 1/clipped_dist)
    #EVOLVE-END
    return 1 / distance_matrix