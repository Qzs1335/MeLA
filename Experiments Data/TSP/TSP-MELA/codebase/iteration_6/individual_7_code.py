import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Convert and validate input
        distance_matrix = np.array(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0 or not np.all(np.isfinite(distance_matrix)):
            raise ValueError("Invalid distance matrix input")
            
        # Safe minimum distance handling with stability factor
        min_nonzero = np.maximum(np.finfo(distance_matrix.dtype).tiny, 
                               np.min(distance_matrix[distance_matrix > 0]))
        stability_factor = np.sqrt(min_nonzero)  # More stable factor
        
        # Replace zeros and negative distances while maintaining the diagonal
        distance_matrix = np.where(distance_matrix <= 0, stability_factor, distance_matrix)
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Calculate dynamic parameters
        dist_mean = np.mean(distance_matrix)
        alpha = max(1.0, np.sqrt(dist_mean + np.finfo(float).eps))
        beta = min(5.0, max(1.0, 2 * alpha))
        
        # Numerical safe transformations
        max_dist = np.max(distance_matrix)
        norm_dist = distance_matrix / (max_dist + np.finfo(float).eps)
        
        # Stable exponential calculations with bounded range
        clamped_vis = np.clip(distance_matrix * beta/alpha, -50, 50)
        visibility = np.exp(-clamped_vis)
        
        # Numerically stable pheromone calculation
        pheromone = 1.0 + np.sin(np.pi * norm_dist)
        pheromone = np.clip(pheromone, np.finfo(float).tiny, None)
        
        # Safe power operations with clipping
        heuristic = np.clip(pheromone**alpha, 1e-50, 1e50) * np.clip(visibility**beta, 1e-50, 1e50)
        
        # Stable softmax calculation
        max_heuristic = np.max(heuristic)
        exp_heuristic = np.exp(np.clip(heuristic - max_heuristic, -50, 50))
        
        softmax = exp_heuristic / (np.sum(exp_heuristic) + np.finfo(float).tiny)
        return np.nan_to_num(softmax, nan=1.0/softmax.size, posinf=1.0/softmax.size, neginf=0.0)
        
    except Exception as e:
        print(f"Heuristic Warning: {str(e)}")
        if not isinstance(distance_matrix, np.ndarray) or distance_matrix.size == 0:
            return None
        softmax = np.ones(distance_matrix.shape)/distance_matrix.size
        return np.nan_to_num(softmax, nan=1.0/softmax.size)
    #EVOLVE-END