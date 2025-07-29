import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        
        # Input validation
        if distance_matrix.size == 0:
            return np.array([])
        if (distance_matrix < 0).any():
            raise ValueError("Negative distances not allowed")
        
        # Safely handle zero and small distances
        np.fill_diagonal(distance_matrix, np.inf)
        min_nonzero = np.min(distance_matrix[distance_matrix > 0])
        max_val = np.max(distance_matrix)
        eps = max(1e-15, min_nonzero/1e6)  # Ultra small but safe epsilon
        
        # Compute visibility and pheromone with numerical safety
        visibility = 1 / np.maximum(distance_matrix, eps)
        log_visibility = np.log10(visibility)  # More stable than natural log
        avg_distance = np.mean(distance_matrix[distance_matrix != np.inf])
        pheromone = np.exp(-distance_matrix/(avg_distance + eps))
        
        # Combine with overflow protection
        heuristic = pheromone * log_visibility 
        
        # Robust normalization
        min_h = np.min(heuristic)
        if np.isinf(min_h) or np.isnan(min_h):
            return np.ones_like(distance_matrix)/distance_matrix.sum(axis=1, keepdims=True)
        
        # Shift to positive range and normalize
        heuristic = np.exp(heuristic - min_h)  # Better for maintaining scale
        heuristic = np.where(np.isfinite(heuristic), heuristic, 0)
        heuristic /= (heuristic.sum(axis=1, keepdims=True) + eps)
        
        # Final validation
        if np.any(np.isnan(heuristic)) or np.any(heuristic < 0):
            return np.ones_like(distance_matrix)/distance_matrix.size
        
        return heuristic
    
    except Exception as e:
        return np.ones_like(distance_matrix)/distance_matrix.size
    #EVOLVE-END