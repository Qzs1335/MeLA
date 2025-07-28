import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Core parameters
        power_base = 0.75
        beta = 0.8
        eps = np.finfo(float).eps * 10
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Robust inverse calculation with smoothing
        inv_dist = np.log1p(1 / (distance_matrix + eps))
        norm_factor = 1 / np.max(inv_dist, initial=1.0)
        heuristic = np.power(inv_dist * norm_factor, beta)
        
        return heuristic / (np.sum(heuristic) + eps)
    
    except Exception as e:
        print(f"Optimized heuristics error: {str(e)}")
        size = len(distance_matrix) if hasattr(distance_matrix, '__len__') else 0
        return np.ones(size)/size if size > 0 else None
    #EVOLVE-END