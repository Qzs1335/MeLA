import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 1.2              # tuned pheromone weight 
        beta = 1.8               # balanced distance weight
        stability_factor = 1e-12 # improved numerical stability
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            return np.ones((1,1))/1  # standard fallback
            
        # Enhanced distance handling
        valid_distances = np.isfinite(distance_matrix) & (distance_matrix > 0)
        if not np.any(valid_distances):
            return np.ones_like(distance_matrix)/distance_matrix.size
            
        distance_matrix = np.where(valid_distances, distance_matrix, np.inf)
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Optimized visibility calculation
        with np.errstate(divide='ignore'):
            visibility = 1/distance_matrix
            visibility = np.nan_to_num(visibility, nan=0.0, posinf=0.0)
            visibility = np.log1p(visibility)  # improved sensitivity
            
        # Efficient heuristic calculation
        heuristic = np.exp(alpha + beta*visibility)
        total = np.sum(heuristic)
        return heuristic/total if total>0 else np.ones_like(heuristic)/heuristic.size
        
    except:
        return np.ones(distance_matrix.shape)/distance_matrix.size if hasattr(distance_matrix, 'shape') else np.ones((1,1))/1
    #EVOLVE-END