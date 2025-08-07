import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-10
    min_dist = np.min(distance_matrix)
    return np.log(1 + (min_dist / (distance_matrix + eps)))
    #EVOLVE-END
    return 1 / distance_matrix