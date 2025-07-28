import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-10
    log_scaled = np.log(1 + distance_matrix)
    nearest_neigh = np.sort(distance_matrix, axis=1)[:,1:4].mean(axis=1)
    return (1 / (distance_matrix + eps)) * (nearest_neigh[:,None] + nearest_neigh) / (2 * log_scaled)
    #EVOLVE-END
    return 1 / distance_matrix