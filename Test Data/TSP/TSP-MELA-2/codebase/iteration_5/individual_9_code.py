import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-10  # Improved numerical stability
    log_scaled = np.cbrt(np.log1p(distance_matrix))  # Smoother gradient
    nearest_neigh = np.partition(distance_matrix, 4, axis=1)[:,1:5]
    neigh_weights = 0.75*np.median(nearest_neigh, axis=1) + 0.25*np.mean(nearest_neigh, axis=1)  # Balanced weighting  
    exp_decay = np.exp(-(distance_matrix**0.8)/(neigh_weights[:,None]+eps))  # Adaptive decay
    return (exp_decay / (distance_matrix + eps)) * np.sqrt(neigh_weights[:,None] + neigh_weights + eps) / log_scaled  
    #EVOLVE-END     
    return 1 / distance_matrix