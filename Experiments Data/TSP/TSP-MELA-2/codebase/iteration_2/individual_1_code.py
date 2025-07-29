import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-10
    log_scaled = np.log(1 + distance_matrix)
    k = max(1, int(np.sqrt(distance_matrix.shape[0])))
    nearest_neigh = np.sort(distance_matrix, axis=1)[:,1:k+1].mean(axis=1)
    exp_factor = np.exp(-distance_matrix/np.median(distance_matrix))
    return (exp_factor/(distance_matrix+eps)) * (nearest_neigh[:,None] + eps) / log_scaled
    #EVOLVE-END
    return 1 / distance_matrix