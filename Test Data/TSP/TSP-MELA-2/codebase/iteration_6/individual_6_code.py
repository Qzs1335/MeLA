import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    nearest_neigh = np.partition(distance_matrix, 2, axis=1)[:,:3] 
    neigh_weights = np.mean(nearest_neigh, axis=1)  
    β = 1/(neigh_weights.mean() + eps)  
    scaled_dist = β * distance_matrix  
    return (1 / (scaled_dist + eps)) * (1 + neigh_weights[:,None]/neigh_weights)
    #EVOLVE-END  
    return 1 / distance_matrix