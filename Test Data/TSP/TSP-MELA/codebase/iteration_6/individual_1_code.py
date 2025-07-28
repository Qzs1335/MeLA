import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Adaptive parameters
        matrix_norm = np.linalg.norm(distance_matrix)
        alpha = 1 + 0.1 * np.log1p(matrix_norm) 
        beta = 2 - 0.1 * np.log1p(matrix_norm)
        stability_factor = max(1e-16, 1e-16 * matrix_norm)
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            raise ValueError("Empty distance matrix")
            
        # Handle zeros and apply exponential decay
        valid_mask = distance_matrix > 0
        distance_matrix = np.where(valid_mask, 
                                 distance_matrix * np.exp(-0.1), 
                                 stability_factor)
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Safer visibility calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            visibility = np.divide(1, distance_matrix, 
                                out=np.zeros_like(distance_matrix),
                                where=valid_mask)
        
        # Numerically stable heuristic
        log_phero = np.log1p(np.ones_like(distance_matrix))
        log_vis = np.log1p(visibility)
        heuristic = np.exp(alpha * log_phero + beta * log_vis)
        
        # Normalize with L1 norm
        heuristic_sum = np.sum(heuristic)
        return heuristic / heuristic_sum if heuristic_sum > 0 else np.ones_like(heuristic)/heuristic.size
        
    except Exception as e:
        print(f"Heuristic Error: {str(e)}")
        return np.ones(distance_matrix.shape)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None
    #EVOLVE-END