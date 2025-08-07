import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    sqrt_dist = np.sqrt(distance_matrix)
    median_neigh = np.median(np.partition(distance_matrix, 5)[:,1:5], axis=1)
    balanced_term = (distance_matrix + median_neigh[:,None]) / (2*sqrt_dist + eps)
    return np.exp(-sqrt_dist/(median_neigh[:,None]+eps)) * balanced_term
    #EVOLVE-END
    return 1 / distance_matrix