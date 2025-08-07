import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-10
    inv_sq_dist = 1 / (distance_matrix**2 + eps)
    log_scale = np.log(1 + distance_matrix/2)
    return inv_sq_dist * (1 + 1/log_scale)
    #EVOLVE-END
    return 1 / distance_matrix