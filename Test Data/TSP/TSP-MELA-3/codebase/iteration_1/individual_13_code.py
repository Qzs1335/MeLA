import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-6
    adjusted_dist = distance_matrix + epsilon
    log_scaled = np.log1p(adjusted_dist)
    return np.exp(-log_scaled**2)
    #EVOLVE-END
    return 1 / distance_matrix