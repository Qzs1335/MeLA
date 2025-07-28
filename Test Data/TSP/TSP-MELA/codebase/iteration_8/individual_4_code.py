import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            return None
            
        eps = 1e-16
        mean_dist = np.mean(distance_matrix[distance_matrix > 0])
        
        # Clipping and safe distance handling
        np.fill_diagonal(distance_matrix, np.inf)  # Prevent self-transitions
        safe_dist = np.clip(distance_matrix, eps, None)
        
        # Adaptive parameter calculations
        alpha = 1 - 0.5 * np.exp(-0.1 * mean_dist)
        beta = 1 + alpha
        
        # Visibility calculation (inverse distance)
        visibility = 1.0 / safe_dist
        visibility = np.nan_to_num(visibility, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Pheromone calculation (sigmoid of normalized distances)
        normalized_dist = distance_matrix / np.clip(mean_dist, eps, None)
        pheromone = 1.0 / (1.0 + np.exp(-np.clip(normalized_dist, -50, 50)))
        
        # Combined heuristic calculation (using log for stability)
        log_pheromone = np.log(pheromone + eps)
        log_visibility = np.log(visibility + eps)
        heuristic = np.exp(alpha * log_pheromone + beta * log_visibility)
        
        # Final normalization with probability distribution constraint
        heuristic = np.nan_to_num(heuristic, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = np.max(heuristic)
        if max_val > eps:
            heuristic /= max_val
        
        # Distance-based scaling constraint
        scaling = 1.0 / (1.0 + np.exp(-np.clip(distance_matrix, -50, 50)))
        result = heuristic * scaling
        
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
    except Exception as e:
        print(f"Heuristic Error: {str(e)}")
        return np.ones(distance_matrix.shape)/distance_matrix.size if distance_matrix.size else None
    #EVOLVE-END