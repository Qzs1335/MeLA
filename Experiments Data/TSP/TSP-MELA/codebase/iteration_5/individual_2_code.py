import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 1 - 0.5*np.exp(-0.1*np.mean(distance_matrix))  # adaptive decay
        beta = 2 + 0.5*np.tanh(np.std(distance_matrix))        # dynamic scaling
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        stability_factor = max(1e-16, np.percentile(distance_matrix, 1)/100)
        
        # Compressed distance processing
        with np.errstate(divide='ignore', invalid='ignore'):
            safe_dist = distance_matrix + stability_factor
            log_dist = np.log(safe_dist)
            visibility = np.exp(-0.5*log_dist)
            
        # Adaptive pheromone
        pheromone = np.ones_like(distance_matrix)
        log_ph = np.clip(np.log(pheromone), -20, 20)
        
        # Dynamic heuristic
        heuristic = np.exp(alpha*log_ph + beta*np.log(visibility + stability_factor))
        return heuristic/np.sum(heuristic) if np.any(heuristic) else np.ones_like(heuristic)/heuristic.size
        
    except Exception as e:
        return np.ones(distance_matrix.shape)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None
    #EVOLVE-END