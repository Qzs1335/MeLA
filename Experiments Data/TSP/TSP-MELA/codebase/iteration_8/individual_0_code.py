import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Dynamic parameters based on matrix properties
        matrix_range = np.ptp(distance_matrix) if distance_matrix.size > 0 else 1
        alpha = 1 - 0.5 * np.exp(-matrix_range)  # Adjustable pheromone weight
        beta = 1 + np.log1p(matrix_range)        # Adaptive distance weight
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            return np.ones(distance_matrix.shape)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None
            
        # Adaptive stability handling
        median_dist = np.median(distance_matrix[distance_matrix > 0])
        stability_factor = max(1e-16, 1e-16 * median_dist) if median_dist > 0 else 1e-16
        
        # Robust visibility calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            safe_dist = np.where(distance_matrix <= 0, stability_factor, distance_matrix)
            visibility = np.exp(-np.log(safe_dist))  # Smoother transformation
            
        # Enhanced pheromone blending
        blended = np.power(distance_matrix + stability_factor, -beta) * np.power(visibility, alpha)
        
        # Stable normalization with softmax
        max_val = np.max(blended)
        if max_val > 0:
            exp_vals = np.exp(blended - max_val)  # Shift for numerical stability
            return exp_vals/np.sum(exp_vals)
        return np.ones_like(blended)/blended.size
        
    except Exception as e:
        return np.ones(distance_matrix.shape)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None
    #EVOLVE-END