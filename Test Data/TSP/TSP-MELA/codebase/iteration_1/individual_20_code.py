import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-6
    normalized_dist = distance_matrix / np.max(distance_matrix)
    log_heuristic = np.log(1 + normalized_dist)
    return 0.5*(1 / (distance_matrix + epsilon)) + 0.5*(1 / (log_heuristic + epsilon))
    #EVOLVE-END
    return 1 / distance_matrix