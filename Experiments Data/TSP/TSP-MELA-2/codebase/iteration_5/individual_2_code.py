import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    safe_dist = distance_matrix + eps
    log_scaled = np.sqrt(np.log(1 + safe_dist))
    nearest_neigh = np.sort(safe_dist, axis=1)[:,1:5]
    neigh_weights = len(nearest_neigh) / np.sum(1/nearest_neigh, axis=1)  # Harmonic mean
    sigmoid_decay = 1/(1+np.exp(safe_dist/neigh_weights[:,None]))  
    linear_hybrid = 0.3/safe_dist + 0.7*(1-sigmoid_decay)
    return linear_hybrid / log_scaled
    #EVOLVE-END
    return 1 / distance_matrix