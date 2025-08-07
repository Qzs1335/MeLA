import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    soft_scaled = np.log1p(distance_matrix)
    nearest_neigh = np.sort(distance_matrix, axis=1)[:,1:5]
    neigh_weights = 1 / np.mean(1/(nearest_neigh + eps), axis=1)
    temp_scaled = (distance_matrix + eps) / neigh_weights[:,None]
    return np.exp(-temp_scaled) * (2 / (soft_scaled + distance_matrix + eps))
    #EVOLVE-END