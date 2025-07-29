import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 0.8              # optimzed pheromone weight
        beta = 1.2               # optimized distance weight
        stability_factor = 1e-16
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            return None
            
        mask = distance_matrix <= 0
        distance_matrix[mask] = stability_factor
        
        # Calculate visibility
        visibility = 1 / np.maximum(distance_matrix, stability_factor)
        
        # Clamped exponential-logarithmic heuristic
        heuristic = np.clip(
            np.exp(alpha*np.log1p(stability_factor) 
                 + beta*(np.log(visibility + stability_factor))),
            stability_factor, 1/stability_factor)
        
        # Non-negative normalization
        return heuristic / np.sum(heuristic) if np.any(heuristic) \
               else np.ones_like(heuristic)/heuristic.size
        
    except Exception as e:
        print(f"Heuristic Error: {str(e)}")
        return None
    #EVOLVE-END