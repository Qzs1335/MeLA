import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            raise ValueError("Empty distance matrix")
            
        # Adaptive stability and scaling
        mean_dist = np.mean(distance_matrix[distance_matrix > 0])
        stability_factor = max(1e-16, mean_dist * 1e-8)
        
        # Exponential distance transform
        scaled_dist = distance_matrix / mean_dist
        visibility = np.exp(-scaled_dist) 
        np.fill_diagonal(visibility, 0)
        
        return visibility / np.sum(visibility)
        
    except Exception as e:
        print(f"Heuristic Error: {str(e)}")
        return np.ones(distance_matrix.shape)/distance_matrix.size if distance_matrix.size else None
    #EVOLVE-END