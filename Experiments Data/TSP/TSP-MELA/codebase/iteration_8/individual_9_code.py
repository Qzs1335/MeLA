import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Dynamic params based on matrix size
        n = len(distance_matrix)
        beta = max(3 - 0.05*n, 1.0)  # Decaying beta encourages more exploration for larger problems
        stability_factor = 1e-16
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            raise ValueError("Empty distance matrix")
            
        # Adaptive zero/inf handling
        distance_matrix[distance_matrix <= 0] = stability_factor
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Smoothed visibility and dynamic weights
        visibility = 1 / np.log(distance_matrix + stability_factor * 10)
        heuristic = visibility ** beta
        
        # Balanced normalization with residual factor
        normalized = heuristic / (np.sum(heuristic) + stability_factor)
        return np.clip(normalized, stability_factor, None)
        
    except Exception as e:
        print(f"Heuristic Error: {str(e)}")
        default = np.ones(distance_matrix.shape)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None
        return default if np.any(np.isfinite(default)) else None
    #EVOLVE-END