import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-10
    log_scaled = 1 + np.log(1 + distance_matrix + eps)
    min_dist = np.min(distance_matrix, axis=1)[:,None]
    
    # Calculate and use standard deviation for normalization
    std_dev = np.std(distance_matrix)
    
    # Adaptive nearest neighbor weighting with std_dev now properly defined
    sigmoid_weight = 1/(1+np.exp(-(distance_matrix/std_dev)))
    dyn_neighbor = (min_dist + sigmoid_weight*distance_matrix.mean(axis=1)[:,None])
    
    return (np.exp(-distance_matrix/5) + 1/(distance_matrix+eps)) * dyn_neighbor / log_scaled
    #EVOLVE-END
    return 1 / (distance_matrix + eps)