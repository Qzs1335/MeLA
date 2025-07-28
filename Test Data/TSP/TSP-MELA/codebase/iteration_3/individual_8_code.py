import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        epsilon = 1e-8
        dist = np.asarray(distance_matrix)
        np.fill_diagonal(dist, np.inf)
        
        # Core log-scale transformation with adaptive term
        log_vis = -np.log(np.maximum(dist, epsilon))
        dynamic_beta = 1 + (np.std(log_vis) / np.mean(log_vis))
        heuristic = np.exp(log_vis * dynamic_beta)
        
        # Fast normalized weighting
        H_sum = np.sum(heuristic)
        return heuristic/(H_sum + epsilon) if H_sum > 0 else np.ones_like(dist)/(dist.size)
        
    except Exception as e:
        print(f"Optimized Error: {str(e)}")
        return np.ones_like(distance_matrix)/(distance_matrix.size or 1)
    #EVOLVE-END