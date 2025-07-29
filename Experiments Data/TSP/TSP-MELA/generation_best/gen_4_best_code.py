import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 1                # pheromone weight 
        beta = 2                 # distance weight
        stability_factor = 1e-16 # numerical stability
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            raise ValueError("Empty distance matrix")
            
        # Handle zeros and infs in distance matrix
        distance_matrix[distance_matrix == 0] = stability_factor
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Stable visibility calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            visibility = 1 / (distance_matrix + stability_factor)
            
        # Handle any remaining infs/nans
        visibility = np.nan_to_num(visibility, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Initial pheromone with stable values
        pheromone = np.ones_like(distance_matrix)
        
        # Stable heuristic calculation
        log_phero = np.log(pheromone + stability_factor)
        log_vis = np.log(visibility + stability_factor)
        heuristic = np.exp(alpha * log_phero + beta * log_vis)
        
        # Stable normalization
        total = np.sum(heuristic)
        if total > 0:
            return heuristic / total
        return np.ones_like(heuristic) / heuristic.size
        
    except Exception as e:
        print(f"Heuristic Error: {str(e)}")
        return np.ones(distance_matrix.shape)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None
    #EVOLVE-END