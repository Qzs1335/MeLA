import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Dynamic weights with tanh decay
        max_iter = np.max(distance_matrix)
        alpha = 5 * np.tanh(0.1/max_iter)  # Adaptive pheromone weight
        beta = 5 * (1 - np.tanh(0.1/max_iter))  # Complement distance weight
        
        eps = max(1e-16, 1e-12 * np.mean(distance_matrix))  # Adaptive stability factor
        distance_matrix = np.clip(distance_matrix, eps, 1e12)
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Stable logarithmic transformations with clipping
        with np.errstate(divide='ignore', invalid='ignore'):
            log_vis = -np.log(distance_matrix + eps)
            log_vis = np.clip(log_vis, -50, 50)
            
        heuristic = np.exp(alpha + beta * log_vis)
        heuristic = heuristic / (np.linalg.norm(heuristic) + eps)
        return np.nan_to_num(heuristic, nan=1.0/heuristic.size)
        
    except Exception as e:
        return np.ones_like(distance_matrix)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None
    #EVOLVE-END