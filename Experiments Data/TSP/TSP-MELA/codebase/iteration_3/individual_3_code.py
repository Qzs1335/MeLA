import numpy as np
import numpy as np

def heuristics_v2(distance_matrix, alpha=1, beta=2):
    #EVOLVE-START
    try:
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            raise ValueError("Empty distance matrix")
        
        if np.any(~np.isfinite(distance_matrix)):
            distance_matrix = np.nan_to_num(distance_matrix, nan=np.inf, posinf=np.inf)
            
        np.fill_diagonal(distance_matrix, np.inf)
        visibility = 1 / np.log1p(distance_matrix)  # log scaling improves stability
        
        heuristic = np.power(visibility, beta) * alpha
        normalized = (heuristic - np.min(heuristic)) / (np.ptp(heuristic) + 1e-10)
        
        return np.nan_to_num(normalized, nan=1/normalized.size)
        
    except:
        uniform = np.ones_like(distance_matrix)/distance_matrix.size
        return np.nan_to_num(uniform, nan=1/uniform.size)
    #EVOLVE-END