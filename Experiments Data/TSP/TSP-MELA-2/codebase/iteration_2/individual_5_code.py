import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-10
    safe_inv = 1/(distance_matrix + eps)
    log_scaled = np.clip(np.log(1 + distance_matrix), 1, 10)
    nearest_neigh = np.sort(distance_matrix, axis=1)[:,1:3].mean(axis=1)
    weights = (0.6*safe_inv) + 0.4*(nearest_neigh[:,None]/log_scaled)
    return weights / weights.max()
    #EVOLVE-END
    return 1 / distance_matrix