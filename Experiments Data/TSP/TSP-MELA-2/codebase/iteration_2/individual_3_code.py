import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-10
    n_closest = max(3, min(5, distance_matrix.shape[1]//4))
    nearest_n = np.partition(distance_matrix, n_closest, axis=1)[:,:n_closest].mean(axis=1)
    log_scale = np.log1p(distance_matrix).clip(-100,100)
    return ((1/(distance_matrix+eps)) * (nearest_n[:,None] + nearest_n) / (2*log_scale))
    #EVOLVE-END
    return 1 / distance_matrix