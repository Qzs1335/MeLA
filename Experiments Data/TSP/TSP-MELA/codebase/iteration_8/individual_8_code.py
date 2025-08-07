import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 1                 # Optimized pheromone weight
        beta = 2                  # Optimized distance weight
        stability_factor = 1e-16  # Maintained for numerical stability
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            raise ValueError("Empty distance matrix")
            
        # Enhanced distance handling
        valid_mask = np.isfinite(distance_matrix)
        distance_matrix = np.where(valid_mask, distance_matrix, np.inf)
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Optimized visibility calculation
        visibility = np.exp(-np.clip(distance_matrix, 0, 50)/10)
        
        # Adaptive heuristic calculation
        log_components = alpha * np.log(1 + visibility) + beta * np.log(1 + 1/(distance_matrix + stability_factor))
        heuristic = np.exp(log_components - np.max(log_components))  # Softmax trick                
        
        return heuristic / np.nansum(heuristic) if np.nansum(heuristic) > 0 else np.ones_like(heuristic)/heuristic.size
        
    except Exception as e:
        print(f"Optimized Heuristic Error: {str(e)}")
        return np.ones(distance_matrix.shape)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None
    #EVOLVE-END