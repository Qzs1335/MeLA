import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Dynamic weights based on matrix characteristics
        mean_dist = np.nanmean(distance_matrix)
        alpha = 1 + np.log1p(mean_dist)/10  # pheromone weight 
        beta = 2 - np.log1p(mean_dist)/10   # distance weight
        stability_factor = 1e-16 * (1 + mean_dist)
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            raise ValueError("Empty distance matrix")
            
        # Exponential smoothing for zero/inf handling
        safe_dist = np.where(distance_matrix <= 0, 
                           np.exp(-20/(distance_matrix.shape[0]+1)),
                           distance_matrix)
        np.fill_diagonal(safe_dist, np.inf)
        
        # Clipped logarithmic visibility
        vis_clip = np.clip(1/(safe_dist+stability_factor), 1e-16, 1e16)
        log_vis = np.log(vis_clip)
        
        # Dimensional pheromone initialization
        size_factor = np.sqrt(distance_matrix.shape[0])
        pheromone = np.ones_like(distance_matrix) * size_factor
        
        # Dimension-aware heuristic
        dim_scale = 1 + np.log(distance_matrix.shape[0])
        heuristic = np.exp((alpha/dim_scale)*np.log(pheromone) + (beta/dim_scale)*log_vis)
        
        # Robust normalization
        norm = np.sum(heuristic, keepdims=True)
        return np.where(norm > 0, heuristic/norm, 1/distance_matrix.size)
        
    except Exception as e:
        return np.ones(distance_matrix.shape)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None
    #EVOLVE-END