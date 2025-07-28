import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Dynamic parameters for exploration-exploitation
        alpha = 1 + np.log1p(distance_matrix.mean())  # adaptive pheromone weight
        beta = 3 - alpha/2                            # complementary distance weight
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            return np.ones_like(distance_matrix)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None
            
        # Safe visibility transform
        safe_dist = distance_matrix.copy()
        np.fill_diagonal(safe_dist, np.nan)
        visibility = 1/np.where(safe_dist > 0, safe_dist, np.nanpercentile(safe_dist, 50))
        
        # Stable pheromone-heuristic synergy
        pheromone = np.ones_like(distance_matrix)
        log_terms = alpha*np.log(pheromone + 1e-16) + beta*np.log(visibility + 1e-16)
        heuristic = np.exp(log_terms - log_terms.max())  # softmax stability
        
        return heuristic / heuristic.sum()
    except Exception:
        return np.ones(distance_matrix.shape)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None
    #EVOLVE-END