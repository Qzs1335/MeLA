import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    normalized_dist = (distance_matrix - np.min(distance_matrix)) / (np.max(distance_matrix) - np.min(distance_matrix) + 1e-10)
    return np.exp(-normalized_dist) + np.log1p(1/(distance_matrix + 1e-10))
    #EVOLVE-END
    return 1 / distance_matrix