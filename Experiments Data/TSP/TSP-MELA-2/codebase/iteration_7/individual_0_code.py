import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    nearest_neigh = np.sort(distance_matrix, axis=1)[:,1]
    neigh_weights = np.median(nearest_neigh)
    return (1 / (distance_matrix + eps)) * (neigh_weights / np.sqrt(nearest_neigh[:,None]))
    #EVOLVE-END
    return 1 / distance_matrix