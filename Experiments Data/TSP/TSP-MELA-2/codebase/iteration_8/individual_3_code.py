import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    log_scaled = np.log(1 + distance_matrix)
    nearest_neigh = np.sort(distance_matrix, axis=1)[:,1:4]  # Smarter neighbor range
    neigh_weights = np.median(nearest_neigh, axis=1)
    exp_component = np.exp(-distance_matrix/np.mean(distance_matrix))
    return exp_component / ((distance_matrix + eps) * np.sqrt(log_scaled))
    #EVOLVE-END
    return 1 / distance_matrix