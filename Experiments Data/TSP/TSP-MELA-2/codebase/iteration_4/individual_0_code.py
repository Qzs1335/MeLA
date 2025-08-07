import numpy as np
import numpy as np  
def heuristics_v2(distance_matrix):  
    #EVOLVE-START  
    eps = 1e-8  
    inv_scaled = 1/(1 + np.sqrt(distance_matrix))  # Sharper gradient  
    nearest_neigh = np.mean(np.sort(distance_matrix, axis=1)[:,1:6], axis=1)  # Faster mean calc  
    scaling = nearest_neigh.mean() / (nearest_neigh[:,None] + eps)  # Adaptive exploration  
    return np.exp(-scaling * distance_matrix) / (distance_matrix + eps) * inv_scaled  
    #EVOLVE-END  
    return 1 / distance_matrix