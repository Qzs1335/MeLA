import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-10
    log_scaled = np.log(1 + distance_matrix)
    nearest_neigh = 1/np.sort(distance_matrix, axis=1)[:,1:4].mean(axis=1)
    weights = np.exp(-distance_matrix/distance_matrix.mean())  # Exponential distance scaling
    exploration = np.random.rand(*distance_matrix.shape)*0.2    # Random exploration factor
    return ((1 / (distance_matrix + eps)) * weights + nearest_neigh[:,None]) / (log_scaled + eps) + exploration
    #EVOLVE-END
    return 1 / distance_matrix